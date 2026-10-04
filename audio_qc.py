"""Quality checks on original waveforms and extracted feature matrices."""

import numpy as np


def waveform_qc(waveform: np.ndarray) -> dict:
    """Measure the decoded waveform without normalizing or changing it."""
    result = {
        "contains_nan": bool(np.isnan(waveform).any()),
        "contains_inf": bool(np.isinf(waveform).any()),
    }
    if waveform.size == 0:
        raise ValueError("Decoded audio is empty.")
    # Use float64 accumulation, without copying the full recording to float64.
    result.update(
        peak_amplitude=float(np.max(np.abs(waveform))),
        mean_amplitude=float(np.mean(waveform, dtype=np.float64)),
        fraction_near_clipping=float(np.count_nonzero(np.abs(waveform) >= 0.99) / waveform.size),
        fraction_exact_zero=float(np.count_nonzero(waveform == 0) / waveform.size),
    )
    squares = np.einsum("ij,ij->", waveform, waveform, dtype=np.float64) if waveform.ndim == 2 else np.einsum("i,i->", waveform, waveform, dtype=np.float64)
    result["rms"] = float(np.sqrt(squares / waveform.size))
    result["dc_offset"] = result["mean_amplitude"]
    for name, value in (("peak_dbfs", result["peak_amplitude"]), ("rms_dbfs", result["rms"])):
        result[name] = float(20 * np.log10(value)) if value > 0 else float("-inf") if value == 0 else float("nan")
    return result


def feature_qc(features, timestamps, duration) -> dict:
    """Validate (frames, dimensions) features and their frame-center times."""
    result = {"feature_mean": None, "feature_std": None, "feature_min": None,
              "feature_max": None, "feature_contains_nan": False,
              "feature_contains_inf": False, "feature_warning": "",
              "feature_error_message": ""}
    try:
        values = np.asarray(features)
        times = np.asarray(timestamps)
        if np.issubdtype(values.dtype, np.number):
            result["feature_contains_nan"] = bool(np.isnan(values).any())
            result["feature_contains_inf"] = bool(np.isinf(values).any())
        if values.ndim != 2 or not all(values.shape):
            raise ValueError("Features must be a nonempty rank-2 (frames, dimensions) array.")
        if not np.issubdtype(values.dtype, np.floating):
            raise ValueError("Features must have a floating-point dtype.")
        if result["feature_contains_nan"] or result["feature_contains_inf"]:
            raise ValueError("Features contain NaN or Inf.")
        result.update(feature_mean=float(np.mean(values, dtype=np.float64)),
                      feature_std=float(np.std(values, dtype=np.float64)),
                      feature_min=float(np.min(values)), feature_max=float(np.max(values)))
        if times.ndim != 1 or len(times) != len(values):
            raise ValueError("Timestamps must be rank-1 with one timestamp per feature frame.")
        if not np.issubdtype(times.dtype, np.floating):
            raise ValueError("Timestamps must have a floating-point dtype.")
        if not np.isfinite(times).all():
            raise ValueError("Timestamps contain NaN or Inf.")
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError("Duration must be finite and positive.")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Timestamps must be strictly increasing.")
        tolerance = max(1e-7, float(duration) * 1e-7)
        if times[0] < -tolerance or times[-1] > duration + tolerance:
            raise ValueError("Timestamps lie outside the processed audio duration.")
        if not np.any(values):
            result["feature_warning"] = "Feature matrix is entirely zero."
        elif result["feature_std"] <= 1e-8:
            result["feature_warning"] = "Feature matrix has near-zero standard deviation (<= 1e-8)."
    except (TypeError, ValueError) as exc:
        result["feature_error_message"] = str(exc)
    return result


def apply_dataset_warnings(rows) -> dict:
    """Append warnings in place; dataset outliers never change processing status.

    Robust scale is 1.4826 * median(abs(x - median(x))). Upper limits use
    max(absolute floor, median + 6 * scale); low RMS uses
    max(-60 dBFS, median - 6 * scale). Duration limits also allow a factor
    of ten around the median, avoiding excessive flags in tiny datasets.
    Existing absolute warnings are retained; dataset clipping and duration
    outliers have separate flags from per-record clipping/duration checks.
    """
    rows = list(rows)
    def bounds(key):
        values = [abs(float(r[key])) if key == "dc_offset" else float(r[key]) for r in rows if r.get("processing_status") != "failed" and r.get(key) is not None and np.isfinite(r[key])]
        if not values:
            return None
        median = float(np.median(values))
        scale = float(1.4826 * np.median(np.abs(np.asarray(values) - median)))
        return median, scale

    thresholds = {"formula": "robust_scale = 1.4826 * MAD; limits use median +/- 6 * robust_scale with absolute floors"}
    for key, floor, name in (("fraction_near_clipping", .001, "clipping"), ("dc_offset", .01, "dc"), ("fraction_exact_zero", .5, "zero_fraction")):
        stats = bounds(key)
        thresholds[name + "_upper"] = max(floor, stats[0] + 6 * stats[1]) if stats else floor
    stats = bounds("rms_dbfs")
    thresholds["rms_dbfs_lower"] = max(-60., stats[0] - 6 * stats[1]) if stats else -60.
    stats = bounds("original_duration_sec")
    thresholds["duration_lower_sec"] = max(.01, min(stats[0] / 10, stats[0] - 6 * stats[1])) if stats else .01
    thresholds["duration_upper_sec"] = max(stats[0] * 10, stats[0] + 6 * stats[1]) if stats else None
    thresholds["duration_formula"] = "lower=max(0.01,min(median/10,median-6*scale)); upper=max(median*10,median+6*scale)"
    for row in rows:
        warnings = row.setdefault("warnings", [])
        for field, flag, limit in (("fraction_near_clipping", "excessive_clipping_warning", "clipping_upper"), ("dc_offset", "dc_warning", "dc_upper"), ("fraction_exact_zero", "zero_fraction_warning", "zero_fraction_upper")):
            value = row.get(field)
            row[flag] = bool(row.get(flag, False) or (value is not None and np.isfinite(value) and abs(value) > thresholds[limit]))
            if row[flag] and flag not in warnings:
                warnings.append(flag)
        value = row.get("rms_dbfs")
        row["low_rms_warning"] = bool(row.get("low_rms_warning", False) or (value is not None and value < thresholds["rms_dbfs_lower"]))
        duration = row.get("original_duration_sec")
        row["duration_outlier_warning"] = bool(duration is not None and (duration < thresholds["duration_lower_sec"] or (thresholds["duration_upper_sec"] is not None and duration > thresholds["duration_upper_sec"])))
        row["clipping_warning"] = bool(row.get("fraction_near_clipping", 0) > 0)
        for flag in ("low_rms_warning", "duration_outlier_warning", "clipping_warning"):
            if row[flag] and flag not in warnings:
                warnings.append(flag)
    return thresholds
