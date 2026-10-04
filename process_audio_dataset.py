"""Preprocess, QC and extract continuous pretrained Dasheng audio sequences."""
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import json
import logging
import platform
import random
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
from audio_io import scan_audio, preprocess_record
from audio_qc import feature_qc, apply_dataset_warnings
from dasheng_extractor import DashengExtractor
from visualization import plot_record


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    # Strict JSON, representing unavailable and nonfinite quantities as null.
    def clean(x):
        if isinstance(x, dict):
            return {k: clean(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [clean(v) for v in x]
        if isinstance(x, float) and not np.isfinite(x):
            return None
        return x
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, ensure_ascii=False), encoding='utf-8')
    tmp.replace(path)


def write_manifest(out, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    tmp = out / 'audio_manifest.csv.tmp'
    with tmp.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, keys)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, ensure_ascii=False) if isinstance(v, list) else v for k, v in row.items()})
    tmp.replace(out / 'audio_manifest.csv')
    write_json(out / 'logs' / 'recording_state.json', rows)


def output_integrity(row, extractor):
    info = sf.info(row['processed_wav_path'])
    if info.samplerate != extractor.sr or info.channels != 1 or info.subtype != 'FLOAT' or info.frames != row['processed_num_samples']:
        return False
    for path_key, hash_key in [('processed_wav_path', 'processed_wav_sha256'), ('feature_path', 'feature_sha256'), ('timestamp_path', 'timestamp_sha256')]:
        if sha256(row[path_key]) != row.get(hash_key):
            return False
    f = np.load(row['feature_path'], mmap_mode='r')
    t = np.load(row['timestamp_path'], mmap_mode='r')
    check = feature_qc(f, t, info.duration)
    return not check['feature_error_message'] and not check['feature_warning'] and f.dtype == np.float32 and f.shape[1] == extractor.model.embed_dim


def summary(rows):
    result = dict(total_audio_files=len(rows),
                  successfully_decoded=sum(r.get('original_num_samples') is not None for r in rows),
                  successfully_resampled=sum(r.get('processed_num_samples', 0) > 0 for r in rows),
                  successfully_feature_extracted=sum(r['processing_status'] in ('success', 'skipped') for r in rows),
                  failed_files=[{'recording_id': r['recording_id'], 'error': r['error_message']} for r in rows if r['processing_status'] not in ('success', 'skipped')],
                  warning_files=[r['recording_id'] for r in rows if r.get('warnings')],
                  sample_rate_distribution=dict(Counter(str(r['original_sample_rate']) for r in rows if r.get('original_sample_rate'))),
                  channel_distribution=dict(Counter(str(r['original_channels']) for r in rows if r.get('original_channels'))),
                  format_distribution=dict(Counter(r['original_format'] for r in rows)))
    durations = [r['original_duration_sec'] for r in rows if r.get('original_duration_sec') is not None]
    for stat, function in [('min', np.min), ('median', np.median), ('mean', np.mean), ('max', np.max)]:
        result['duration_' + stat] = float(function(durations)) if durations else None
    result['total_audio_hours'] = sum(durations) / 3600
    for key, flag in [('clipping', 'clipping_warning'), ('low_rms', 'low_rms_warning'), ('large_dc_offset', 'dc_warning'),
                      ('duration_mismatch', 'duration_mismatch_warning'), ('feature_nan', 'feature_contains_nan'),
                      ('feature_inf', 'feature_contains_inf'), ('large_zero_fraction', 'zero_fraction_warning'), ('feature_errors', 'feature_warning')]:
        result['files_with_' + key] = [r['recording_id'] for r in rows if r.get(flag)]
    dims = sorted({r['dasheng_feature_dim'] for r in rows if r.get('dasheng_feature_dim')})
    result['dasheng_feature_dim'] = dims[0] if len(dims) == 1 else dims
    rates = [r['dasheng_token_rate_hz'] for r in rows if r.get('dasheng_token_rate_hz')]
    for stat, function in [('mean', np.mean), ('min', np.min), ('max', np.max)]:
        result['audio_token_rate_' + stat] = float(function(rates)) if rates else None
    result['total_audio_tokens'] = sum(r.get('dasheng_num_tokens', 0) for r in rows)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--qc_samples', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--chunk_seconds', type=float)
    parser.add_argument('--checkpoint', type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.qc_samples < 0:
        parser.error('workers >= 1 and qc_samples >= 0 required')
    inp, out = args.input_dir.resolve(), args.output_dir.resolve()
    if inp == out or inp in out.parents or out in inp.parents:
        parser.error('Input and output directories must be separate, without nesting.')
    files = scan_audio(inp)
    for directory in ['wav_16k_mono', 'dasheng_features', 'dasheng_timestamps', 'qc', 'logs']:
        (out / directory).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s',
                        handlers=[logging.FileHandler(out / 'logs' / 'processing.log', encoding='utf-8'), logging.StreamHandler()])
    errors = logging.FileHandler(out / 'logs' / 'errors.log', encoding='utf-8')
    errors.setLevel(logging.ERROR)
    logging.getLogger().addHandler(errors)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(min(4, torch.get_num_threads()))
    versions = {}
    for name in ['torch', 'torchaudio', 'transformers', 'numpy', 'soundfile', 'soxr', 'dasheng', 'einops', 'matplotlib']:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    source_hashes = {p.name: sha256(p) for p in Path(__file__).parent.glob('*.py')}
    settings = dict(seed=args.seed, chunk_seconds=args.chunk_seconds, package_versions=versions,
                    source_code_sha256=source_hashes, checkpoint_sha256=sha256(args.checkpoint) if args.checkpoint else None)
    signature = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
    previous = {}
    config_path = out / 'processing_config.json'
    if config_path.exists() and (out / 'logs' / 'recording_state.json').exists():
        old_config = json.loads(config_path.read_text(encoding='utf-8'))
        if old_config.get('run_signature') == signature:
            previous = {r['relative_path']: r for r in json.loads((out / 'logs' / 'recording_state.json').read_text(encoding='utf-8'))}
    try:
        ffmpeg = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, check=True).stdout.splitlines()[0]
    except (OSError, subprocess.CalledProcessError):
        ffmpeg = None
    config = dict(processing_timestamp=datetime.now(timezone.utc).isoformat(), python_version=platform.python_version(),
                  input_directory=str(inp), output_directory=str(out), target_sample_rate=16000,
                  channel_strategy='arithmetic mean of original channels', resampling_library='soxr', resampling_settings={'quality': 'HQ'},
                  ffmpeg_version=ffmpeg, run_signature=signature, **settings)
    try:
        extractor = DashengExtractor(args.device, args.checkpoint, args.chunk_seconds)
        config.update(extractor.config)
        distribution = metadata.distribution('dasheng').read_text('direct_url.json')
        config['dasheng_implementation_revision'] = json.loads(distribution) if distribution else versions['dasheng']
    except Exception as exc:
        extractor = None
        model_error = f'Dasheng initialization failed: {type(exc).__name__}: {exc}'
        logging.exception(model_error)
        config['model_error'] = model_error
    write_json(config_path, config)
    counts = Counter(p.stem for p in files)
    used_ids = set()
    jobs, rows = [], []
    for source in files:
        rel = source.relative_to(inp)
        rid = source.stem
        if counts[rid] > 1 or rid in used_ids:
            rid += '__' + hashlib.sha256(rel.as_posix().encode()).hexdigest()[:16]
        if rid in used_ids:
            raise RuntimeError(f'Recording ID collision for {rel}')
        used_ids.add(rid)
        row = dict(recording_id=rid, filename=source.name, relative_path=rel.as_posix(), format=source.suffix.lstrip('.'),
                   file_size_bytes=source.stat().st_size, original_path=str(source),
                   processed_wav_path=str(out / 'wav_16k_mono' / rel.parent / (rid + '.wav')),
                   feature_path=str(out / 'dasheng_features' / rel.parent / (rid + '.npy')),
                   timestamp_path=str(out / 'dasheng_timestamps' / rel.parent / (rid + '.npy')),
                   dasheng_model_name='Dasheng-Base', feature_warning='', error_message='')
        old = previous.get(rel.as_posix())
        try:
            if not args.overwrite and old and extractor and old['recording_id'] == rid and old.get('original_sha256') == sha256(source) and output_integrity(old, extractor):
                row.update(old)
                row.update(processing_status='skipped', warnings=[])
                rows.append(row)
                logging.info('Skip valid outputs: %s', rid)
                continue
        except Exception:
            logging.info('Reprocessing incomplete or invalid outputs: %s', rid)
        rows.append(row)
        jobs.append(row)
    def preprocess(row):
        return preprocess_record(Path(row['original_path']), Path(row['processed_wav_path']))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for row, result in zip(jobs, pool.map(preprocess, jobs)):
            row.update(result)
            logging.info('Preprocessing %s: %s', row['recording_id'], row['processing_status'])
    thresholds = apply_dataset_warnings(rows)
    config['qc_thresholds'] = thresholds
    for row in rows:
        if row.get('processing_status') == 'skipped':
            continue
        if row.get('processing_status') == 'preprocessed':
            try:
                if extractor is None:
                    raise RuntimeError(model_error)
                waveform, sr = sf.read(row['processed_wav_path'], dtype='float32')
                if sr != extractor.sr:
                    raise RuntimeError('Processed sample rate differs from Dasheng sample rate.')
                features, times, extraction = extractor.extract(waveform)
                row.update(extraction)
                row.update(feature_qc(features, times, row['processed_duration_sec']))
                if row['feature_error_message']:
                    row['feature_warning'] = row['feature_error_message']
                    raise RuntimeError(row['feature_error_message'])
                for key, value in [('feature_path', features), ('timestamp_path', times)]:
                    target = Path(row[key])
                    target.parent.mkdir(parents=True, exist_ok=True)
                    tmp = target.with_suffix('.npy.tmp')
                    with tmp.open('wb') as stream:
                        np.save(stream, value, allow_pickle=False)
                    tmp.replace(target)
                row.update(dasheng_feature_dim=features.shape[1], dasheng_num_tokens=len(features),
                           dasheng_token_rate_hz=extractor.config['native_token_rate_hz'],
                           dasheng_first_timestamp=float(times[0]), dasheng_last_timestamp=float(times[-1]), processing_status='success')
                for path_key, hash_key in [('processed_wav_path', 'processed_wav_sha256'), ('feature_path', 'feature_sha256'), ('timestamp_path', 'timestamp_sha256')]:
                    row[hash_key] = sha256(row[path_key])
                if row['feature_warning']:
                    row['warnings'].append(row['feature_warning'])
                logging.info('Features %s: %s, timestamps %.6f..%.6f sec', row['recording_id'], features.shape, times[0], times[-1])
            except Exception as exc:
                row.update(processing_status='feature_failed', error_message=f'{type(exc).__name__}: {exc}', feature_warning=str(exc))
                row['warnings'].append(str(exc))
                logging.exception('Feature failure: %s', row['recording_id'])
        if row.get('error_message'):
            logging.error('%s: %s', row['recording_id'], row['error_message'])
        write_manifest(out, rows)
    good = [r for r in rows if r['processing_status'] in ('success', 'skipped')]
    chosen = random.Random(args.seed).sample(good, min(len(good), args.qc_samples if len(files) >= 20 else len(good)))
    for row in chosen:
        target = out / 'qc' / Path(row['relative_path']).parent / (row['recording_id'] + '_qc.png')
        try:
            if args.overwrite or row['processing_status'] != 'skipped' or not target.exists():
                plot_record(row, target)
            row['qc_figure_path'] = str(target)
        except Exception as exc:
            row['warnings'].append(f'QC figure failed: {exc}')
            logging.exception('QC figure: %s', row['recording_id'])
    config['qc_selected_recording_ids'] = [r['recording_id'] for r in chosen]
    # Verify every original file is byte-identical after the full pipeline.
    for row in rows:
        row['original_unchanged'] = sha256(row['original_path']) == row.get('original_sha256')
        if not row['original_unchanged']:
            row['warnings'].append('Original file hash changed during processing.')
    examples = []
    for row in random.Random(args.seed).sample(good, min(3, len(good))):
        features = np.load(row['feature_path'], mmap_mode='r')
        times = np.load(row['timestamp_path'])
        wave_info = sf.info(row['processed_wav_path'])
        example = dict(recording_id=row['recording_id'], duration=row['processed_duration_sec'], waveform_shape=[wave_info.frames],
                       feature_shape=list(features.shape), first_5_timestamps=times[:5].tolist(), last_5_timestamps=times[-5:].tolist(),
                       timestamp_range=[float(times[0]), float(times[-1])], feature_mean=float(features.mean(dtype=np.float64)), feature_std=float(features.std(dtype=np.float64)))
        examples.append(example)
        print(json.dumps(example, indent=2))
    write_json(out / 'qc' / 'sample_integrity_checks.json', examples)
    write_manifest(out, rows)
    result = summary(rows)
    result['all_original_files_unchanged'] = all(r['original_unchanged'] for r in rows)
    result['qc_figures_generated'] = sum(bool(r.get('qc_figure_path')) for r in rows)
    write_json(out / 'qc_summary.json', result)
    write_json(config_path, config)
    print(json.dumps(result, indent=2))
    return 1 if result['failed_files'] else 0


if __name__ == '__main__':
    import os
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    raise SystemExit(main())
