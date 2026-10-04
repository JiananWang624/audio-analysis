"""Native temporal Dasheng features; no pooling or feature interpolation."""
from pathlib import Path
import hashlib

import numpy as np
import torch
import torch.nn.functional as F
from dasheng import dasheng_base
from dasheng.pretrained.pretrained import PRETRAINED_CHECKPOINTS


class DashengExtractor:
    def __init__(self, device='auto', checkpoint=None, chunk_seconds=None):
        self.device = 'cuda' if device == 'auto' and torch.cuda.is_available() else ('cpu' if device == 'auto' else device)
        if self.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA requested but unavailable; use --device cpu.')
        if checkpoint:
            path = Path(checkpoint).resolve()
        else:
            url = PRETRAINED_CHECKPOINTS['dasheng_base']
            torch.hub.load_state_dict_from_url(url, map_location='cpu')
            path = Path(torch.hub.get_dir()) / 'checkpoints' / 'dasheng_base.pt'
        self.model = dasheng_base(path=str(path)).eval().to(self.device)
        self.model.requires_grad_(False)
        mel = self.model.front_end[0]
        stft = mel.spectrogram
        conv = self.model.patch_embed.proj
        self.sr = mel.sample_rate
        self.hop = stft.hop_length
        self.nfft = stft.n_fft
        self.center = stft.center
        self.stft_pad = stft.pad
        self.kernel = conv.kernel_size[-1]
        self.stride = conv.stride[-1]
        if conv.padding != (0, 0) or conv.dilation != (1, 1):
            raise RuntimeError('Unsupported Dasheng convolution timing geometry.')
        freq_tokens = (self.model.n_mels - conv.kernel_size[0]) // conv.stride[0] + 1
        if freq_tokens != 1 or self.model.pooling == 'token' or self.kernel != self.stride:
            raise RuntimeError('Expected nonoverlapping temporal patches covering the full frequency axis, without CLS tokens.')
        self.max_tokens = self.model.time_pos_embed.shape[-1]
        self.chunk_tokens = self.max_tokens if chunk_seconds is None else min(self.max_tokens, int(chunk_seconds * self.sr / (self.hop * self.stride)))
        if self.chunk_tokens < 1:
            raise ValueError('--chunk_seconds must cover at least one native token.')
        with path.open('rb') as stream:
            checksum = hashlib.file_digest(stream, 'sha256').hexdigest()
        self.config = dict(
            dasheng_model_identifier='Dasheng-Base', model_checkpoint_url=PRETRAINED_CHECKPOINTS['dasheng_base'],
            model_checkpoint_path=str(path), model_checkpoint_sha256=checksum,
            feature_dimension=self.model.embed_dim, sample_rate=self.sr,
            hop_samples=self.hop, n_fft=self.nfft, win_samples=stft.win_length,
            stft_center=self.center, stft_pad=self.stft_pad,
            patch_size=list(conv.kernel_size), patch_stride=list(conv.stride),
            temporal_stride_sec=self.hop * self.stride / self.sr,
            native_token_rate_hz=self.sr / (self.hop * self.stride), device=self.device, dtype='float32',
            chunk_tokens=self.chunk_tokens, chunk_seconds=self.chunk_tokens * self.hop * self.stride / self.sr,
            chunk_overlap_seconds=0,
            stitching_strategy='Full-recording official frontend and patch embedding; sequential official forward_features blocks; concatenate each token once.',
            timestamp_definition='Center of local input STFT patch, not full Transformer attention receptive field. For the final padded patch, center of its real support intersected with the recording.',
            tail_policy='Replicate up to patch_size-1 log-mel frames at the end for a final partial token; no audio is removed or padded in the saved WAV.',
        )

    @torch.no_grad()
    def extract(self, waveform):
        n = len(waveform)
        if n == 0 or not np.isfinite(waveform).all():
            raise ValueError('Empty or nonfinite waveform.')
        x = torch.from_numpy(np.asarray(waveform, dtype=np.float32)).unsqueeze(0).to(self.device)
        # Reflect STFT requires > n_fft/2 samples. Padding is inference-only.
        min_samples = self.nfft // 2 + 1 if self.center else self.nfft
        audio_padding = max(0, min_samples - n)
        if audio_padding:
            x = F.pad(x, (0, audio_padding), mode='replicate')
        spec = self.model.forward_to_spec(x)
        if audio_padding and self.center:
            spec = spec[..., :n // self.hop + 1]
        frames = spec.shape[-1]
        padding = (-frames) % self.kernel
        if padding:
            spec = F.pad(spec, (0, padding, 0, 0), mode='replicate')
        patches = self.model.patch_embed(spec)
        total = patches.shape[-1]
        results = []
        for start in range(0, total, self.chunk_tokens):
            result = self.model.forward_features(patches[..., start:start + self.chunk_tokens])
            if result.shape != (1, min(self.chunk_tokens, total - start), self.model.embed_dim):
                raise RuntimeError(f'Unexpected Dasheng output shape: {tuple(result.shape)}')
            results.append(result[0].cpu().numpy())
        features = np.concatenate(results).astype(np.float32, copy=False)
        # Centered STFT frame j is at j*hop; uncentered frame j at j*hop+n_fft/2.
        # A temporal convolution patch covers kernel STFT frames. No linspace.
        offset = (0 if self.center else self.nfft / 2) - self.stft_pad
        starts = np.arange(total, dtype=np.float64) * self.stride
        timestamps = (offset + (starts + (self.kernel - 1) / 2) * self.hop) / self.sr
        duration = n / self.sr
        if padding or audio_padding:
            # Only the partial terminal patch uses its unpadded, real support.
            real_end = min((frames - 1) * self.hop + offset, n - 1)
            timestamps[-1] = (min(starts[-1] * self.hop + offset, n - 1) + real_end) / (2 * self.sr)
        valid = (timestamps >= 0) & (timestamps <= duration)
        # Inference-only padding of extremely short audio may create extra tokens.
        features, timestamps = features[valid], timestamps[valid]
        if not len(timestamps) or np.any(np.diff(timestamps) <= 0):
            raise RuntimeError('Invalid native token timestamps.')
        return features, timestamps, dict(model_input_padding_samples=audio_padding, feature_tail_padding_frames=padding,
                                         dasheng_effective_token_rate_hz=len(features) / duration)
