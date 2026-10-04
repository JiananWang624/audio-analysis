# Robot-assisted feeding: audio processing handoff

Audio preprocessing and pretrained Dasheng-Base temporal feature extraction are complete for 19 of 20 discovered recordings; one source contains no audio samples.

| Current state | Count |
|---|---:|
| Total discovered recordings | 20 |
| Successfully processed / downstream-usable | 19 |
| Usable without warnings | 14 |
| Usable with warnings | 5 |
| Failed / unusable | 1 |

**Evidence boundary:** “Implemented pipeline behavior” below describes the inspected implementation and recorded configuration. “Actual observed dataset result” describes existing outputs checked on 2026-10-04. This handoff was created without rerunning preprocessing, model inference or tests.

## 1. Research purpose and processing rationale

These representations are intended for feeding-stage recognition and bite-timing / time-to-next-bite prediction, later combined with face landmarks, head pose, gaze, video and other temporal features. Audio is retained as a continuous recording-level temporal sequence, rather than one embedding per recording or per clip.

The conservative preprocessing is intended to preserve acoustic events and original timing: chewing, swallowing, mouth sounds, speech, breathing, pauses, utensils, robot sounds and environmental sounds may all be relevant. The goal is not clean speech. Whether any of these cues actually help the research tasks is **Unknown / not verified**; no downstream experiment has been performed here.

## 2. Locations and source audio — actual observed dataset result

Workspace root: `F:/code/feeding_data_analysis/audio_analysis/`.
All relative paths in this document resolve from that root. Existing CSV paths are absolute Windows paths; remap them if transferring the workspace.

| Source property | Observed value |
|---|---|
| Input directory | `F:/code/feeding_data_analysis/audio_analysis/audio/` |
| Recordings / formats | 20 / all WAV |
| Sample-rate distribution | 48,000 Hz: 20 |
| Channel distribution | Mono: 20; stereo: 0 |
| Total original duration | 10,290.752 seconds = 2.8585422222 hours |
| Duration range, including empty source | 0–1,303.765333 seconds |
| Nonempty original duration range | 131.946667–1,303.765333 seconds |
| Decoder recorded in manifest | soundfile: 20; ffmpeg: 0 |

Source provenance before these WAV files, including whether they were extracted from video, is **Unknown / not verified**. All 20 current original-file SHA256 hashes match the manifest. The pipeline also recorded `original_unchanged=True` for every source.

## 3. Audio preprocessing

**Implemented pipeline behavior:** recursively discover supported audio extensions; decode with soundfile into float32; use ffprobe/ffmpeg if soundfile cannot decode; compute QC on the original decoded channels; average channels; resample with `soxr`, quality `HQ`, to 16 kHz; save a derived mono WAV with subtype `FLOAT`. No gain adjustment is applied. Only resampling's required anti-alias filtering is introduced at waveform preprocessing.

**Actual observed dataset result:** all sources were already mono, so channel averaging did not mix distinct channels. The 19 nonempty recordings were decoded with soundfile and resampled from 48 kHz to 16 kHz. The 19 saved working WAVs were inspected: sample rate 16,000 Hz, one channel, WAV `FLOAT` (32-bit float), and frame counts matching the manifest. Maximum absolute processed-minus-original duration difference is approximately `0.000020833333` seconds (0.020833 ms), below the 20 ms warning tolerance.

Working copies: `processed_audio/wav_16k_mono/<recording_id>.wav`.
Original recordings remain under `audio/`; derived files do not replace them.

The inspected processing path performs **no denoising, speech enhancement, VAD, silence removal, temporal trimming, time stretching, amplitude/loudness normalization, dynamic range compression, AGC, arbitrary waveform filtering, or global/clip-level temporal feature pooling**. It does not resample features to a multimodal frame rate. Dasheng's own mel/dB frontend is part of the pretrained model, not waveform normalization. Inference-only terminal spectrogram padding is documented in Section 6.

## 4. QC procedure and actual thresholds

**Implemented pipeline behavior:** measure original waveform peak amplitude/dBFS, RMS/dBFS, mean amplitude, signed DC offset, near-clipping fraction, exact-zero fraction, NaN/Inf, sample count and duration. Record decode/empty/nonfinite failures. Compare processed and original sample-derived durations. Feature QC checks matrix shape, nonempty dimensions, finite values, nonzero values and variance, plus timestamp shape, finiteness, ordering and range; scalar feature mean/std/min/max are recorded.

Dataset thresholds use median `m` and scale `s = 1.4826 * median(abs(x - m))`, excluding failed rows and unavailable/nonfinite values. Absolute per-record flags are retained even if the dataset-derived upper limit is larger.

| Check | Actual implemented rule / threshold for this dataset |
|---|---|
| Near-clipping sample | `abs(x) >= 0.99` |
| `clipping_warning` | At least one near-clipping sample |
| `excessive_clipping_warning` | Near-clipping fraction > `max(0.001, m+6s)`; recorded limit `0.001` (0.1%) |
| `low_rms_warning` | RMS dBFS < `max(-60, m-6s)`; recorded limit `-60 dBFS` |
| `dc_warning` | Absolute DC > 0.01, or > dataset limit `max(0.01, m+6s)`; recorded limit `0.01` |
| `zero_fraction_warning` | Exact-zero fraction > 0.5, or > dataset limit `max(0.5, m+6s)`; recorded limit `0.5` |
| `duration_warning` / `duration_mismatch_warning` | Absolute processed-minus-original duration > `0.020` seconds |
| `container_duration_warning` | For ffmpeg decoding, container versus decoded duration differs by > `0.020` seconds; unused here |
| `duration_outlier_warning` | Duration < `max(0.01,min(m/10,m-6s))` or > `max(10m,m+6s)`; recorded bounds `0.01` and `4479.146667` seconds |
| Feature zero / variance warning | Entire matrix zero, or aggregate feature std <= `1e-8` |
| Timestamp range tolerance in feature QC | `max(1e-7, processed_duration_sec * 1e-7)` seconds |

Near-clipping is a threshold flag, not proof of audible clipping. Warnings retain recordings; unreadable, empty/zero-duration or nonfinite waveforms cannot proceed. A feature error is recorded per recording and processing continues for other recordings.

## 5. Actual QC results and downstream usability

| Observed outcome | Count |
|---|---:|
| Near-clipping warnings | 5 |
| Excessive-clipping warnings | 0 |
| Low-RMS / DC-offset / large-zero-fraction warnings | 0 / 0 / 0 |
| Processed-duration mismatch / container-duration warning | 0 / 0 |
| Original waveform NaN / Inf flags | 0 / 0 |
| Decoder failures / decoder fallback warnings | 0 / 0 |
| Zero-length audio failures | 1 |
| Duration-outlier warnings | 1, on the same zero-length source |
| Feature extraction errors / feature warnings | 0 / 0 |
| Feature NaN / Inf / timestamp errors | 0 / 0 / 0 |
| Recordings with any warning, including failed source | 6 |

### Usable with warnings

All five have only `clipping_warning`; no hard failure. Their current status is `skipped`, meaning successful outputs were reused on the second run. Recommendation: retain for multimodal dataset construction. The warning extent is very small; perceptual severity is **Unknown / not verified** because listening was not performed.

| recording_id | Near-clipping samples | Fraction of samples (%) | Severity / use |
|---|---:|---:|---|
| `trial_0010_audio` | 1 | 0.0000022246 | Very sparse threshold exceedance; retain |
| `trial_0015_audio` | 2 | 0.0000194903 | Very sparse threshold exceedance; retain |
| `trial_003-def_audio` | 1 | 0.0000015979 | Very sparse threshold exceedance; retain |
| `trial_004-s_audio` | 1 | 0.0000043115 | Very sparse threshold exceedance; retain |
| `trial_008_audio` | 3 | 0.0000063708 | Very sparse threshold exceedance; retain |

Combined: 8 near-clipping samples, approximately `0.0000016196%` of all valid original waveform samples. Recorded peaks for these files are 0.99054–0.99359; the flag alone does not establish hard saturation.

### Unusable recording — DO NOT use in downstream audio modeling

- **ID:** `trial_0021_audio`.
- **Original path:** `F:/code/feeding_data_analysis/audio_analysis/audio/trial_0021_audio.wav`.
- **Failure stage:** decoded-waveform validation, before resampling or inference.
- **Observed reason:** WAV file is 44 bytes, with a valid 48 kHz mono PCM16 header and a zero-length `data` chunk; zero samples, zero seconds.
- **Recorded error:** `ValueError: Decoded audio is empty.`; status `failed`; warning `duration_outlier_warning`.
- **Outputs:** no working WAV, feature matrix or timestamp array. CSV output-path fields are intended destinations and do not mean files exist.
- **Cause during capture/export:** **Unknown / not verified**. Original is preserved. The two error-log entries represent first run and rerun of this same file, not two failed recordings.

### Explicit usability rule

Use manifest rows with `processing_status` in `{"success", "skipped"}`, no `error_message` or `feature_error_message`, existing WAV/feature/timestamp files, a finite nonempty float32 `[T,768]` feature matrix, and finite float64 `[T]` timestamps with matching length, strict ordering and valid recording bounds. All 19 retained recordings currently meet these conditions. The 14 rows with an empty JSON `warnings` list are usable without warnings; the other five above are usable with warnings. This establishes audio artifact usability, not completed audio-video synchronization or proven predictive utility.

## 6. Encoder, feature structure, timing and chunking

### Implemented pipeline behavior and recorded configuration

| Property | Recorded value |
|---|---|
| Model | Generic pretrained `Dasheng-Base` |
| Implementation | `https://github.com/RicherMans/Dasheng.git` |
| Implementation revision | `91bc15c728f08e8d10f741311d8afc42fb33d497` (`dasheng` 0.0.9) |
| Checkpoint source | `https://zenodo.org/records/11511780/files/dasheng_base.pt?download=1` |
| Local checkpoint | `model_cache/dasheng_base.pt` |
| Checkpoint SHA256 | `533784d2bfc026e21dde4562fb129d2d4911028fb2c0932615ea0b7b6278481f` |
| Execution | CUDA, float32; `model.eval()`, `requires_grad_(False)`, `torch.no_grad()` |
| Training / fine-tuning | None |
| Native feature dimension / stride | 768 / 0.040 seconds (25 Hz) |
| STFT | 16 kHz; FFT/window 512 samples; hop 160 samples; centered; explicit pad 0 |
| Patch kernel / stride | `[64,4]` / `[64,4]`; one patch across the frequency axis |
| Transformer context | 252 temporal tokens, approximately 10.08 seconds |
| Chunk overlap | 0 seconds |

The complete-recording frontend and patch embedding are evaluated before splitting Transformer inference into consecutive contexts. Each context's output is appended once, with its original global patch index; there are no overlapping boundary tokens to deduplicate and no waveform/STFT restart at chunk boundaries. Saved outputs are one continuous timeline per recording, not independent clips. All 19 nonempty recordings are longer than one context and therefore use chunked Transformer inference. Attention context changes at block boundaries; this is not full-recording attention.

For complete patch index `k`:

```text
STFT frame j center = j * hop / sample_rate
token_time[k] = (k * patch_stride + (patch_kernel - 1)/2) * hop / sample_rate
              = 0.015 + 0.040*k seconds for this recorded configuration
```

These are local acoustic patch-center timestamps, not centers of the entire bidirectional attention receptive field. They are not generated with `linspace`.

**Terminal extension:** the official complete-patch path would omit an incomplete terminal patch. This implementation replicates up to three terminal normalized log-mel frames to produce a final partial token. Its timestamp is the midpoint of the real frame-center span, bounded by the recording's last sample: `(start_frame_sample + min(last_real_frame_sample, N-1)) / (2*sample_rate)`. This is an explicit extension; the final token and its attention context need not equal the unextended official output. The saved WAV is neither padded nor trimmed. Extremely short audio may need internal waveform padding for STFT; its count is recorded separately.

### Actual observed dataset result

- 19 feature files: float32 `[T_audio,768]`, with `T_audio` varying from 3,299 to 32,595; total 257,279 tokens.
- 19 timestamp files: float64 `[T_audio]`, seconds relative to decoded recording start (`t=0`).
- All feature matrices are finite, nonempty, nonzero, have nonzero variance and match recorded dimensions/token counts.
- All timestamps match feature length, are finite and strictly increasing, and fall within actual working-WAV duration without needing range tolerance.
- Complete patches have the native 40 ms stride; partial final-token spacing may differ. `dasheng_token_rate_hz` records nominal 25 Hz; `dasheng_effective_token_rate_hz` is `T / processed_duration_sec`, including endpoint effects.
- 14 recordings have `feature_tail_padding_frames` between 1 and 3; five have 0. All 19 have `model_input_padding_samples=0`.
- Example: `trial_0013_audio` has waveform `[5810517]`, features `[9079,768]`, timestamps `[9079]`, range `0.015–363.135` seconds; working duration `363.1573125` seconds. Three seeded examples with first/last timestamps and feature statistics are saved in `processed_audio/qc/sample_integrity_checks.json`.

## 7. Outputs, consistency and manifest schema

Output root: `F:/code/feeding_data_analysis/audio_analysis/processed_audio/`.

```text
processed_audio/
├── wav_16k_mono/          # 19 derived FLOAT WAV files
├── dasheng_features/     # 19 recording_id.npy float32 temporal matrices
├── dasheng_timestamps/   # 19 recording_id.npy float64 time arrays
├── qc/                  # 19 three-panel PNGs + sample_integrity_checks.json
├── logs/
│   ├── processing.log   # initial extraction and subsequent skip events
│   ├── errors.log       # two entries, same zero-length source
│   └── recording_state.json  # internal rerun state
├── audio_manifest.csv
├── qc_summary.json
└── processing_config.json
```

Current outputs are flat because source files are flat. Implementation preserves relative subdirectories and adds relative-path hash suffixes to duplicate stems. QC PNGs show a waveform min/max envelope, display-only log-mel spectrogram and native token L2 norm; all horizontal axes use seconds. Display resolution is independent of saved features.

**Actual consistency check:** manifest row count and summary total both equal 20. Usable manifest rows, summary feature-success count, actual WAV count, actual feature count and actual timestamp count all equal 19. All 19 feature files have exactly one corresponding timestamp file with matching length; no extra output arrays or working WAVs were found. All raw and derived-file SHA256 hashes match the CSV; working-WAV metadata and actual array shapes/dtypes match the CSV. No artifact/count inconsistency was found.

**Counter semantics:** summary `successfully_decoded=20` includes the readable zero-length WAV because the implementation counts rows with known original sample counts, even when that count is zero. There are 19 nonempty decoded waveforms. Summary `warning_files` contains 6 IDs, including the failed zero-duration file; usable-with-warning is 5. These are definition differences, not unexplained file-count mismatches. Latest CSV statuses are 19 `skipped` and 1 `failed`; `skipped` means valid previously computed outputs, not missing processing.

`audio_manifest.csv` is the primary dataset index (UTF-8 BOM). Important **existing** fields:

| Group | CSV fields |
|---|---|
| Identity / paths | `recording_id`, `filename`, `relative_path`, `original_path`, `processed_wav_path`, `feature_path`, `timestamp_path`, `qc_figure_path` |
| Original metadata | `original_format`, `original_sample_rate`, `original_channels`, `original_num_samples`, `original_duration_sec`, `bit_depth`, `file_size_bytes`, `decoder` |
| Working audio | `processed_sample_rate`, `processed_channels`, `processed_num_samples`, `processed_duration_sec`, `duration_difference_sec` |
| Waveform QC | `peak_amplitude`, `peak_dbfs`, `rms`, `rms_dbfs`, `mean_amplitude`, `dc_offset`, `fraction_near_clipping`, `fraction_exact_zero`, `contains_nan`, `contains_inf` |
| Features / timing | `dasheng_model_name`, `dasheng_feature_dim`, `dasheng_num_tokens`, `dasheng_token_rate_hz`, `dasheng_effective_token_rate_hz`, `dasheng_first_timestamp`, `dasheng_last_timestamp`, `feature_mean`, `feature_std`, `feature_min`, `feature_max`, `feature_tail_padding_frames`, `model_input_padding_samples` |
| Status / warnings | `processing_status`, `warnings`, `decode_warning`, `clipping_warning`, `excessive_clipping_warning`, `low_rms_warning`, `dc_warning`, `zero_fraction_warning`, `duration_warning`, `duration_mismatch_warning`, `duration_outlier_warning`, `container_duration_warning`, `feature_warning`, `feature_error_message`, `feature_contains_nan`, `feature_contains_inf`, `error_message` |
| Provenance / integrity | `original_sha256`, `processed_wav_sha256`, `feature_sha256`, `timestamp_sha256`, `original_unchanged` |

With `csv.DictReader`, values are strings: parse `warnings` as JSON and booleans explicitly (`value == "True"`); `bool("False")` is incorrect. Missing output metadata on failed rows is empty, not evidence of success. `feature_warning` is a message string, unlike boolean waveform warning fields.

## 8. Loading and reproducibility

Read existing outputs; preprocessing does not need to be rerun for alignment:

```python
import csv
import numpy as np

with open("processed_audio/audio_manifest.csv", encoding="utf-8-sig", newline="") as f:
    usable = [r for r in csv.DictReader(f)
              if r["processing_status"] in {"success", "skipped"}
              and not r["error_message"] and not r["feature_error_message"]]
row = usable[0]  # includes the five retained warning recordings
features = np.load(row["feature_path"], mmap_mode="r", allow_pickle=False)      # [T,768]
timestamps = np.load(row["timestamp_path"], mmap_mode="r", allow_pickle=False)  # [T]
```

`features[i]` corresponds to `timestamps[i]` seconds from this audio recording's start. Preserve these explicit timestamps when joining other modalities; do not infer time solely from array index or reinterpret the partial final token as a regular grid endpoint.

Recorded environment: Python 3.13.9; torch/torchaudio `2.11.0+cu128`; numpy 2.3.5; soundfile 0.14.0; soxr 1.1.0; einops 0.8.2; matplotlib 3.10.7. Transformers was not installed or used. Seed 42 is recorded for deterministic selection; model eval/no-grad and deterministic execution settings are in the implementation. Exact model revision, checkpoint hash, source hashes, package versions and ffmpeg version are recorded in `processing_config.json`; checkpoint revision beyond its URL and SHA256 is **Unknown / not verified**.

Recorded configuration timestamp: `2026-10-04T10:58:39.368076+00:00` (11:58:39 Europe/London). This is the rerun timestamp, not the original extraction start. Logs show original feature extraction and the rerun skipping 19 existing valid outputs.

Entry point: `process_audio_dataset.py`; supporting files: `audio_io.py`, `audio_qc.py`, `dasheng_extractor.py`, `visualization.py`; dependency pins: `requirements.txt`; usage: `README.txt`. A project-local `.venv` exists. If deliberate reprocessing is required:

```powershell
.\.venv\Scripts\python.exe process_audio_dataset.py --input_dir audio --output_dir processed_audio --device auto --workers 2 --checkpoint model_cache/dasheng_base.pt
```

`--overwrite` forces recalculation. Rerun skipping requires matching recorded processing signature/input hash and intact output hashes/metadata/arrays. CPU preprocessing uses threads; model inference uses one process. Exit code 1 is expected while the empty source remains in the dataset. Do not interpret it as 19 failed extractions. Synthetic tests exist in `test_pipeline.py`; they were not rerun for this documentation task, and no standalone saved test report was found.

## 9. Caveats and work not completed

- **Audio/video synchronization:** timestamps are audio-relative. They do not prove absolute audio-video synchronization. Container extraction provenance, capture offsets and clock drift are **Unknown / not verified**; obtain corresponding evidence before declaring modalities aligned.
- **Future context:** Dasheng has bidirectional attention within approximately 10.08-second contexts, plus centered STFT. A token at time `t` can use later audio. These features are not established as causal online features; avoid future-information leakage in prospective bite prediction. Changing chunk length changes representations.
- **Semantics / confounding:** embeddings are generic pretrained acoustic representations, not chewing, swallowing or readiness labels. Robot motors, utensils, speech and room sounds can contribute alongside human eating sounds. High stage-prediction performance would not alone prove recognition of human behavior; inspect modality leakage and confounding later.
- **Temporal limitations:** tokens preserve the encoder's native temporal structure, not every waveform-level detail. Chunk attention boundaries and terminal partial-patch extension are part of the representation. Human inspection/listening of all QC results and task-specific cue usefulness are **Unknown / not verified**.
- **Warning scope:** five sparse near-clipping flags do not make those recordings unusable. The empty recording remains excluded regardless of its intended destination paths.

Not completed: audio-video temporal alignment; unified multimodal timeline; feeding-stage label construction; time-to-next-bite label construction; multimodal fusion; temporal classifier or bite-timing model; subject-independent train/validation/test split; audio-contribution or leakage ablations. No face landmarks, head pose, gaze or video data were loaded by this audio pipeline. Their availability/status outside this workspace is **Unknown / not verified**.

## 10. Recommended next step

Build the multimodal temporal dataset, starting from `processed_audio/audio_manifest.csv`: establish audio-video offsets/clock conventions, combine usable audio features/timestamps with landmarks, head pose, gaze and video/other timestamps on a common timeline, then construct feeding-stage and time-to-next-bite labels. Preserve native audio arrays as the source representation. No classifier architecture is specified here.

## Compact state for next agent

```yaml
audio_processing_complete: true
completion_scope: "19 valid recordings processed; 1 empty source retained as failure"
raw_audio_modified: false
raw_hashes_rechecked: true
workspace_root: "F:/code/feeding_data_analysis/audio_analysis"
input_directory: "audio"
output_directory: "processed_audio"
target_sample_rate_hz: 16000
processed_channels: 1
working_wav_subtype: "FLOAT"
audio_encoder: "Dasheng-Base"
encoder_implementation_revision: "91bc15c728f08e8d10f741311d8afc42fb33d497"
feature_dimension: 768
feature_dtype: "float32"
feature_structure: "[T, D]"
native_token_rate_hz: 25.0
native_token_stride_ms: 40.0
total_audio_tokens: 257279
timestamps_available: true
timestamp_dtype: "float64"
timestamp_unit: "seconds"
timestamp_origin: "decoded audio recording start"
timestamp_definition: "local acoustic patch center; final partial-patch exception"
audio_video_synchronization_verified: false
causal_online_features_verified: false
chunk_seconds: 10.08
chunk_overlap_seconds: 0
terminal_spectrogram_padding_used: true
total_recordings: 20
readable_sources_including_empty: 20
nonempty_decoded_recordings: 19
usable_recordings: 19
usable_without_warning: 14
usable_with_warning: 5
all_warning_recordings_including_unusable: 6
unusable_recordings: 1
unusable_recording_ids: ["trial_0021_audio"]
usable_warning_recording_ids:
  - "trial_0010_audio"
  - "trial_0015_audio"
  - "trial_003-def_audio"
  - "trial_004-s_audio"
  - "trial_008_audio"
warning_type: "clipping_warning (near-clipping sample threshold)"
current_usable_manifest_status: "skipped"
accepted_usable_manifest_statuses: ["success", "skipped"]
actual_wav_files: 19
actual_feature_files: 19
actual_timestamp_files: 19
actual_qc_figures: 19
artifact_consistency_issues: []
manifest: "processed_audio/audio_manifest.csv"
qc_summary: "processed_audio/qc_summary.json"
processing_config: "processed_audio/processing_config.json"
next_step: "multimodal temporal alignment and dataset construction"
```
