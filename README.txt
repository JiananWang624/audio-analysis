Audio preprocessing and native temporal Dasheng-Base features

Run from this directory using the isolated environment:
  .venv\Scripts\python.exe process_audio_dataset.py --input_dir audio --output_dir processed_audio --device auto --workers 2 --checkpoint model_cache/dasheng_base.pt

Install into a fresh virtual environment with:
  python -m pip install -r requirements.txt
Requirements target the CUDA 12.8 Windows environment used here. CPU inference
is available with --device cpu. Without --checkpoint the official Base checkpoint
is downloaded to the standard torch cache. Its SHA256 is stored in the run config.

Optional arguments: --overwrite, --workers, --qc_samples, --seed,
--chunk_seconds (defaults to the official positional context capacity).
A recording failure is retained in the manifest; all other recordings continue.
Exit code 1 means at least one recording failed, including zero-duration sources.

Outputs are one float32 [T,D] .npy matrix and float64 [T] timestamp .npy per
recording, with a FLOAT 16kHz mono working WAV. Relative subfolders are preserved.
Duplicate stems receive a deterministic relative-path hash suffix.
Use np.load(path, mmap_mode='r') for feature batch loading.
Timestamps are seconds from that recording's decoded start, not an externally
synchronized video clock. Any audio/video start offset must be provided later.

Timing derivation (values read from actual pretrained model):
Centered STFT frame j has time j*hop/sample_rate. A patch spanning frames
k*stride ... k*stride+kernel-1 has center
  (k*stride + (kernel-1)/2)*hop/sample_rate.
For this checkpoint: 160-sample hop, 4-frame nonoverlapping patches, 16kHz;
complete patch centers start at .015 seconds and advance by .040 seconds (25Hz).
No linspace timing and no temporal feature pooling or feature resampling are used.

The frontend is evaluated on the complete recording, preserving official STFT
and AmplitudeToDB behavior (including its per-recording top_db). Patch embedding
is applied globally before Transformer contexts are split into at most the
official positional capacity. Outputs concatenate directly, with no duplicate
tokens, overlap or re-started STFT frame grid. The default complete-patch output
has been checked against official full-recording inference.

The official implementation omits incomplete terminal patches. To retain the
tail, we replicate at most three terminal log-mel frames for one partial token;
its timestamp is the center of real support within the recording. The tail
padding count is stored per recording. This is an explicit inference extension;
it does not pad or trim the saved waveform. Extremely short inputs are padded
only internally to satisfy the frontend STFT, and that padding is also recorded.

Timestamps identify the local acoustic patch, not the entire attention receptive
field: Dasheng uses bidirectional attention inside each approximately 10.08s
default context. Future online bite prediction must account for this look-ahead
to avoid future-information leakage. Changing --chunk_seconds changes attention
context and therefore representations, and is recorded for reproducibility.
Dasheng's pretrained mel frontend is part of the model; it is not preprocessing
denoising or waveform amplitude normalization.

QC warning thresholds use dataset medians and 1.4826*MAD with documented absolute
floors, saved in processing_config.json. Any original sample abs >= .99 is flagged
for clipping. Low volume, DC and exact zeros are warnings; silence is retained.
NaN/Inf waveforms, unreadable, empty and zero-duration audio are failures.
The original file SHA256 is recorded and checked again at the end.

Re-running skips valid outputs when input hashes, processing signature, output
hashes, WAV metadata and feature/timestamp integrity agree. --overwrite forces
processing. A changed implementation/configuration invalidates the prior run.
CPU decoding/resampling uses bounded threads; model inference uses one process.

QC figures show the waveform min/max envelope, display-only log-mel spectrogram,
and native feature norm. All horizontal axes use recording seconds. They do not
classify chewing, swallowing or other acoustic events.

Reports:
  processed_audio/audio_manifest.csv
  processed_audio/qc_summary.json
  processed_audio/processing_config.json
  processed_audio/qc/sample_integrity_checks.json (three seeded random examples)
  processed_audio/logs/processing.log and errors.log

Focused synthetic validation:
  .venv\Scripts\python.exe -m unittest test_pipeline -v

Official sources:
https://github.com/RicherMans/Dasheng
https://zenodo.org/records/11511780
