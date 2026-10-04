"""Focused synthetic regression checks; no experimental audio is modified."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from audio_io import preprocess_record
from audio_qc import apply_dataset_warnings, feature_qc
from dasheng_extractor import DashengExtractor


class AudioTests(unittest.TestCase):
    def test_amplitude_duration_and_silence(self):
        with tempfile.TemporaryDirectory() as folder:
            source, target = Path(folder) / 'raw.wav', Path(folder) / 'out.wav'
            x = np.ones((48000, 2), dtype=np.float32) * [.1, .3]
            sf.write(source, x, 48000, subtype='FLOAT')
            row = preprocess_record(source, target)
            y, sr = sf.read(target)
            self.assertEqual(row['original_channels'], 2)
            self.assertEqual((sr, len(y)), (16000, 16000))
            np.testing.assert_allclose(y[100:-100], .2, atol=1e-5)
            self.assertLess(abs(row['duration_difference_sec']), 1e-5)
            sf.write(source, np.zeros(48000), 48000, subtype='FLOAT')
            silent = preprocess_record(source, target)
            self.assertEqual(silent['processing_status'], 'preprocessed')
            apply_dataset_warnings([silent])
            self.assertTrue(silent['low_rms_warning'])
            self.assertTrue(silent['zero_fraction_warning'])

    def test_invalid_features(self):
        f = np.ones((4, 3), dtype=np.float32)
        self.assertTrue(feature_qc(f, np.array([.1, .2, .2, .4]), 1)['feature_error_message'])
        f[0, 0] = np.nan
        self.assertTrue(feature_qc(f, np.arange(4, dtype=float), 4)['feature_contains_nan'])


class DashengTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.extractor = DashengExtractor('auto', Path('model_cache/dasheng_base.pt'))

    def test_matches_official_long_inference(self):
        e = self.extractor
        # Exactly two complete official contexts: no terminal partial patch.
        count = (e.max_tokens * e.kernel * 2 - 1) * e.hop
        wave = np.random.default_rng(42).normal(0, .03, count).astype(np.float32)
        f, t, _ = e.extract(wave)
        with torch.no_grad():
            reference = e.model(torch.from_numpy(wave)[None].to(e.device))[0].cpu().numpy()
        np.testing.assert_allclose(f, reference, rtol=1e-5, atol=1e-5)
        self.assertEqual(f.shape, (e.max_tokens * 2, e.model.embed_dim))
        self.assertAlmostEqual(t[0], (e.kernel - 1) * e.hop / (2 * e.sr))
        np.testing.assert_allclose(np.diff(t), e.hop * e.stride / e.sr, atol=1e-12)

    def test_partial_patch_short_audio_and_stitching(self):
        e = self.extractor
        for count in [1, 100, 16000, e.sr * 11 + 123]:
            f, t, _ = e.extract(np.zeros(count, dtype=np.float32))
            self.assertFalse(feature_qc(f, t, count / e.sr)['feature_error_message'])
            self.assertEqual(len(f), int(np.ceil((count // e.hop + 1) / e.kernel)))
            self.assertTrue(np.all(t >= 0))
            self.assertTrue(np.all(t <= count / e.sr))
            self.assertTrue(np.all(np.diff(t) > 0))


if __name__ == '__main__':
    unittest.main()
