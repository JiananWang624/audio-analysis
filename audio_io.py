"""Decode original audio and save 16 kHz mono FLOAT WAV model inputs."""

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf
import soxr

from audio_qc import waveform_qc


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".ogg", ".oga", ".opus", ".m4a", ".aac", ".aiff", ".aif", ".aifc", ".wma", ".mp4", ".caf", ".au", ".amr", ".ac3", ".ape", ".webm"}


def scan_audio(root: Path) -> list[Path]:
    """Return supported files recursively, in stable relative-path order."""
    root = Path(root)
    if not root.is_dir():
        raise ValueError(f"Audio directory does not exist: {root}")
    return sorted((p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS), key=lambda p: p.relative_to(root).as_posix())


def _decode(source: Path, row: dict) -> np.ndarray:
    try:
        info = sf.info(source)
        row.update(original_sample_rate=info.samplerate, original_channels=info.channels,
                   original_num_samples=info.frames, original_duration_sec=info.duration,
                   original_format=info.format, decoder="soundfile")
        if info.subtype in {"PCM_U8", "PCM_S8", "PCM_16", "PCM_24", "PCM_32", "FLOAT", "DOUBLE"}:
            row["bit_depth"] = {"PCM_U8": 8, "PCM_S8": 8, "PCM_16": 16, "PCM_24": 24, "PCM_32": 32, "FLOAT": 32, "DOUBLE": 64}[info.subtype]
        audio, rate = sf.read(source, dtype="float32", always_2d=True)
        row["original_sample_rate"] = rate
        return audio
    except (sf.LibsndfileError, RuntimeError) as exc:
        row["warnings"].append(f"soundfile decoding failed; using ffmpeg: {exc}")
        row["decode_warning"] = True
    row["decoder"] = "ffmpeg"
    try:
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a:0", "-show_streams", "-show_format", "-of", "json", str(source)], check=True, capture_output=True, text=True)
        metadata = json.loads(probe.stdout)
        if not metadata.get("streams"):
            raise ValueError("ffprobe found no audio stream.")
        stream = metadata["streams"][0]
        rate, channels = int(stream["sample_rate"]), int(stream["channels"])
        row.update(original_sample_rate=rate, original_channels=channels,
                   original_format=metadata.get("format", {}).get("format_name", source.suffix.lstrip(".")))
        for key in ("bits_per_raw_sample", "bits_per_sample"):
            bits = int(stream.get(key) or 0)
            if bits and stream.get("codec_name", "").startswith(("pcm_", "flac", "alac")):
                row["bit_depth"] = bits
                break
        declared_duration = stream.get("duration", metadata.get("format", {}).get("duration"))
        if declared_duration is not None:
            row["container_duration_sec"] = float(declared_duration)
            row["original_duration_sec"] = float(declared_duration)
        decoded = subprocess.run(["ffmpeg", "-v", "error", "-i", str(source), "-map", "0:a:0", "-vn", "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"], check=True, capture_output=True)
        if len(decoded.stdout) % (4 * channels):
            raise ValueError("ffmpeg returned an incomplete interleaved audio frame.")
        return np.frombuffer(decoded.stdout, dtype="<f4").reshape(-1, channels)
    except FileNotFoundError as exc:
        raise RuntimeError("Audio needs ffmpeg decoding; install ffmpeg and ffprobe and add them to PATH.") from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else exc.stderr
        raise RuntimeError(f"{Path(exc.cmd[0]).name} failed: {stderr.strip()}") from exc


def preprocess_record(source: Path, wav_path: Path) -> dict:
    """Return a flat metadata/QC record, including failures, for one source."""
    source, wav_path = Path(source), Path(wav_path)
    row = {"original_sha256": None, "original_sample_rate": None, "original_channels": None,
           "original_num_samples": None, "original_duration_sec": None, "bit_depth": None,
           "original_format": source.suffix.lstrip("."), "decoder": None, "warnings": [],
           "contains_nan": False, "contains_inf": False, "decode_warning": False,
           "duration_mismatch_warning": False, "container_duration_warning": False,
           "duration_difference_sec": None, "clipping_warning": False,
           "low_rms_warning": False, "dc_warning": False, "zero_fraction_warning": False,
           "duration_warning": False, "processing_status": "failed", "error_message": ""}
    try:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        row["original_sha256"] = digest.hexdigest()
        if source.stat().st_size == 0:
            raise ValueError("Source audio file is empty (0 bytes); provide a valid audio recording.")
        waveform = _decode(source, row)
        row.update(original_num_samples=len(waveform), original_channels=waveform.shape[1],
                   original_duration_sec=len(waveform) / row["original_sample_rate"])
        row.update(contains_nan=bool(np.isnan(waveform).any()), contains_inf=bool(np.isinf(waveform).any()))
        if waveform.size == 0:
            raise ValueError("Decoded audio is empty.")
        row.update(waveform_qc(waveform))
        if row["contains_nan"] or row["contains_inf"]:
            raise ValueError("Decoded audio contains NaN or Inf.")
        row["clipping_warning"] = row["fraction_near_clipping"] > 0
        row["low_rms_warning"] = row["rms_dbfs"] < -60
        row["zero_fraction_warning"] = row["fraction_exact_zero"] > .5
        row["dc_warning"] = abs(row["dc_offset"]) > .01
        for flag in ("clipping_warning", "low_rms_warning", "zero_fraction_warning", "dc_warning"):
            if row[flag]:
                row["warnings"].append(flag)
        declared = row.get("container_duration_sec")
        row["container_duration_warning"] = bool(declared is not None and abs(declared - row["original_duration_sec"]) > .020)
        if row["container_duration_warning"]:
            row["warnings"].append("Container duration differs from decoded sample duration.")
        mono = waveform.mean(axis=1, dtype=np.float32)
        processed = soxr.resample(mono, row["original_sample_rate"], 16000, quality="HQ")
        if not len(processed) or not np.isfinite(processed).all():
            raise ValueError("Resampling produced empty or nonfinite audio.")
        wav_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(wav_path, processed, 16000, subtype="FLOAT", format="WAV")
        row.update(processed_sample_rate=16000, processed_channels=1,
                   processed_num_samples=len(processed), processed_duration_sec=len(processed) / 16000,
                   processing_status="preprocessed")
        # Subtract integer sample counts before division to preserve the boundary.
        row["duration_difference_sec"] = (len(processed) * row["original_sample_rate"] - len(waveform) * 16000) / (16000 * row["original_sample_rate"])
        row["duration_warning"] = abs(row["duration_difference_sec"]) > .020
        row["duration_mismatch_warning"] = row["duration_warning"]
        if row["duration_warning"]:
            row["warnings"].append("Processed duration differs from original by more than 0.020 seconds.")
    except Exception as exc:
        row["error_message"] = f"{type(exc).__name__}: {exc}"
    return row
