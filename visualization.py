"""Three-panel, recording-time QC figures (visualization only)."""
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_record(row, path: Path):
    wave, sr = sf.read(row['processed_wav_path'], dtype='float32')
    features = np.load(row['feature_path'], mmap_mode='r')
    times = np.load(row['timestamp_path'])
    # A coarser display STFT keeps long-recording figures compact. Saved model
    # inputs and native features are never downsampled for this visualization.
    hop = max(160, int(len(wave) / 4000))
    with torch.no_grad():
        mel = torchaudio.transforms.MelSpectrogram(sample_rate=sr, n_fft=512,
                                                  hop_length=hop, n_mels=64)(torch.from_numpy(wave))
        mel = 10 * torch.log10(mel.clamp_min(1e-10))
    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    step = max(1, len(wave) // 20000)
    # Display a min/max envelope so isolated transients are not lost by decimation.
    count = len(wave) // step
    blocks = wave[:count * step].reshape(count, step)
    wt = (np.arange(count) * step + step / 2) / sr
    axes[0].fill_between(wt, blocks.min(1), blocks.max(1), linewidth=0, color='steelblue')
    axes[0].set_ylabel('Amplitude')
    axes[1].pcolormesh(np.arange(mel.shape[-1]) * hop / sr, np.arange(64), mel.numpy(), shading='auto', cmap='magma')
    axes[1].set_ylabel('Mel bin (dB)')
    axes[2].plot(times, np.linalg.norm(features, axis=1), linewidth=.6)
    axes[2].set_ylabel('Token L2 norm')
    axes[2].set_xlabel('Time (seconds from recording start)')
    axes[2].set_xlim(0, len(wave) / sr)
    fig.suptitle(row['recording_id'])
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
