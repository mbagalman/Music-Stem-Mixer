import copy
import math
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import stem_mixer_oop as mixer


def base_config():
    return {
        "sample_rate": 48000,
        "stem_detection": {
            "vocals": ["vox", "vocal"],
            "drums": ["drum"],
            "bass": ["bass"],
        },
        "alignment": {"enabled": False, "threshold_dbfs": -45.0, "min_offset_ms": 25.0},
        "chains": {
            "default": {
                "hpf_hz": 50.0,
                "stereo_width": 1.0,
                "compressor": {
                    "threshold_db": -20.0,
                    "ratio": 2.0,
                    "attack_ms": 15.0,
                    "release_ms": 120.0,
                },
                "eq_peaks": [],
                "parallel_comp_blend": 0.0,
            },
            "vocals": {
                "hpf_hz": 80.0,
                "stereo_width": 1.0,
                "deesser": {"mode": "builtin", "amount": 0.8, "frequency_hz": 6000.0, "ratio": 4.0},
                "compressor": {
                    "threshold_db": -18.0,
                    "ratio": 2.5,
                    "attack_ms": 10.0,
                    "release_ms": 80.0,
                },
                "eq_peaks": [],
                "parallel_comp_blend": 0.0,
            },
        },
        "buses": {
            "default": {"target_rms_dbfs": -20.0, "gain_db": 0.0, "pan": 0.0, "stereo_width": 1.0},
            "vocals": {"target_rms_dbfs": -18.0, "gain_db": 0.0, "pan": 0.0, "stereo_width": 1.0},
        },
        "sidechain": {"enabled": False},
        "master": {
            "hpf_hz": 30.0,
            "compressor": {
                "threshold_db": -14.0,
                "ratio": 1.4,
                "attack_ms": 15.0,
                "release_ms": 150.0,
            },
            "limiter": {"ceiling_dbfs": -1.0, "release_ms": 100.0},
            "target_lufs": -14.0,
            "reference_match": {"tonal_balance_mode": "off", "amount": 0.0, "max_tilt_db_per_octave": 3.0},
            "multiband": {"enabled": False},
        },
    }


class FakeStem:
    def __init__(self, role, audio):
        self.role = role
        self.audio = audio
        self.path = Path(f"{role}.wav")
        self.input_stats = {
            "input_rms_dbfs": mixer.calc_rms_dbfs(audio),
            "input_peak_dbfs": mixer.calc_peak_dbfs(audio),
            "leading_silence_ms": 0.0,
            "silent": False,
        }
        self.stats = {}
        self.diagnostics = []


class StemMixerTests(unittest.TestCase):
    def test_normalize_config_migrates_deprecated_fields(self):
        cfg = base_config()
        cfg["chains"]["vocals"]["target_rms_dbfs"] = -19.0
        cfg["chains"]["vocals"]["external_deesser_vst3"] = "plugin.vst3"
        del cfg["buses"]["vocals"]["target_rms_dbfs"]
        cfg["master"]["limiter_ceil_dbfs"] = -1.5

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            normalized = mixer.normalize_config(copy.deepcopy(cfg))

        self.assertEqual(normalized["buses"]["vocals"]["target_rms_dbfs"], -19.0)
        self.assertEqual(normalized["chains"]["vocals"]["deesser"]["mode"], "external")
        self.assertEqual(normalized["chains"]["vocals"]["deesser"]["external_deesser_vst3"], "plugin.vst3")
        self.assertEqual(normalized["master"]["limiter"]["ceiling_dbfs"], -1.5)
        self.assertGreaterEqual(len(caught), 3)

    def test_validate_config_rejects_invalid_multiband(self):
        cfg = base_config()
        cfg["master"]["multiband"] = {
            "enabled": True,
            "bands": [],
            "ratios": [1.5],
            "threshold_db": -20.0,
            "attack_ms": 10.0,
            "release_ms": 120.0,
        }
        with self.assertRaises(ValueError):
            mixer.validate_config(cfg)

    def test_peak_limiter_enforces_ceiling(self):
        y = np.vstack([
            np.array([0.0, 2.0, -2.0, 0.5], dtype=np.float32),
            np.array([0.0, -1.5, 1.5, -0.25], dtype=np.float32),
        ])
        limited = mixer.apply_peak_limiter(y, 48000, -1.0, 100.0)
        self.assertLessEqual(np.max(np.abs(limited)), mixer.db_to_amp(-1.0) + 1e-5)

    def test_builtin_deesser_reduces_high_band_energy(self):
        sr = 48000
        t = np.arange(sr // 4) / sr
        low = 0.2 * np.sin(2 * math.pi * 220 * t)
        high = 0.9 * np.sin(2 * math.pi * 8000 * t)
        signal = np.vstack([low + high, low + high]).astype(np.float32)
        cfg = {
            "mode": "builtin",
            "frequency_hz": 5000.0,
            "threshold_db": -30.0,
            "ratio": 6.0,
            "amount": 1.0,
            "attack_ms": 1.0,
            "release_ms": 80.0,
        }
        before_high = mixer.split_low_high(signal, 5000.0, sr)[1]
        after = mixer.apply_builtin_deesser(signal, sr, cfg)
        after_high = mixer.split_low_high(after, 5000.0, sr)[1]
        self.assertLess(np.max(np.abs(after_high)), np.max(np.abs(before_high)))

    def test_bus_normalization_is_role_level(self):
        tone = np.sin(2 * math.pi * 440 * (np.arange(4800) / 48000.0)).astype(np.float32)
        audio = np.vstack([tone, tone])
        cfg = {"target_rms_dbfs": -20.0, "gain_db": 0.0, "pan": 0.0, "stereo_width": 1.0}
        one = mixer.Bus("guitar", [FakeStem("guitar", audio)], cfg, 48000)
        two = mixer.Bus("guitar", [FakeStem("guitar", audio), FakeStem("guitar", audio)], cfg, 48000)
        one.process()
        two.process()
        self.assertAlmostEqual(one.stats["post_rms_dbfs"], two.stats["post_rms_dbfs"], places=2)

    def test_integration_smoke_with_synthetic_audio(self):
        cfg = base_config()
        cfg["sidechain"] = {
            "enabled": True,
            "trigger": "drums",
            "targets": ["bass"],
            "amount": 0.2,
            "attack_ms": 5.0,
            "release_ms": 80.0,
            "mode": "low_band",
            "low_band_hz": 150.0,
        }
        cfg["buses"]["bass"] = {
            "target_rms_dbfs": -19.0,
            "gain_db": 0.0,
            "pan": 0.0,
            "stereo_width": 1.0,
            "mono_below_hz": 150.0,
        }

        sr = 48000
        t = np.arange(sr // 8) / sr
        library = {
            "song_vocal.wav": np.vstack([0.2 * np.sin(2 * math.pi * 440 * t), 0.2 * np.sin(2 * math.pi * 440 * t)]).astype(np.float32),
            "song_drum.wav": np.vstack([0.4 * np.sin(2 * math.pi * 60 * t), 0.4 * np.sin(2 * math.pi * 60 * t)]).astype(np.float32),
            "song_bass.wav": np.vstack([0.35 * np.sin(2 * math.pi * 80 * t), 0.35 * np.sin(2 * math.pi * 80 * t)]).astype(np.float32),
        }

        def fake_read(path, target_sr):
            return library[Path(path).name], target_sr

        with mock.patch.object(mixer, "read_audio_file", side_effect=fake_read), \
            mock.patch.object(mixer, "run_board", side_effect=lambda audio, board, sr, chunk: audio), \
            mock.patch.object(mixer, "Compressor", side_effect=lambda **kwargs: object()), \
            mock.patch.object(mixer, "HighpassFilter", side_effect=lambda *args, **kwargs: object()), \
            mock.patch.object(mixer, "HighShelfFilter", side_effect=lambda *args, **kwargs: object()), \
            mock.patch.object(mixer, "PeakFilter", side_effect=lambda *args, **kwargs: object()):
            instance = mixer.Mixer(
                files=[Path("song_vocal.wav"), Path("song_drum.wav"), Path("song_bass.wav")],
                cfg=cfg,
                sr_cli=None,
                chunk_samples=4096,
                reference=None,
            )
            instance.process_stems()
            instance.build_buses()
            instance.apply_sidechain()
            instance.render_mix()
            instance.master_chain()

        self.assertIsNotNone(instance.mix)
        self.assertIn("master_limiter", instance.report)
        self.assertIn("vocals", instance.buses)
        self.assertLessEqual(np.max(np.abs(instance.mix)), mixer.db_to_amp(-1.0) + 1e-4)


if __name__ == "__main__":
    unittest.main()
