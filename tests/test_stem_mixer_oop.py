import copy
import math
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest import mock

import numpy as np

import stem_mixer_oop as mixer

try:
    import soundfile as sf
    HAVE_SOUNDFILE_FOR_TESTS = True
except Exception:
    sf = None
    HAVE_SOUNDFILE_FOR_TESTS = False


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

        # 1 s of audio — long enough to exceed pyloudnorm's 400 ms block size when installed.
        sr = 48000
        t = np.arange(sr) / sr
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


@unittest.skipUnless(HAVE_SOUNDFILE_FOR_TESTS, "soundfile not installed")
class AudioLoadTests(unittest.TestCase):
    """Real on-disk WAV round-trip — no mocks of soundfile or stereoify."""

    def setUp(self):
        self.sr = 48000
        self.t = np.arange(self.sr // 4) / self.sr
        self.tone = np.sin(2 * math.pi * 440 * self.t).astype(np.float32)

    def _write(self, path: Path, data: np.ndarray) -> None:
        # soundfile expects (frames,) for mono and (frames, channels) for multi.
        sf.write(str(path), data, self.sr, subtype="FLOAT")

    def test_read_audio_file_loads_mono_as_stereo(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "mono.wav"
            self._write(path, self.tone)
            audio, sr = mixer.read_audio_file(path, self.sr)
        self.assertEqual(sr, self.sr)
        self.assertEqual(audio.shape, (2, self.tone.shape[0]))
        np.testing.assert_allclose(audio[0], audio[1], atol=1e-6)
        np.testing.assert_allclose(audio[0], self.tone, atol=1e-5)

    def test_read_audio_file_preserves_stereo(self):
        stereo = np.vstack([self.tone, -self.tone]).T  # (frames, 2)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "stereo.wav"
            self._write(path, stereo)
            audio, sr = mixer.read_audio_file(path, self.sr)
        self.assertEqual(sr, self.sr)
        self.assertEqual(audio.shape, (2, self.tone.shape[0]))
        np.testing.assert_allclose(audio[0], self.tone, atol=1e-5)
        np.testing.assert_allclose(audio[1], -self.tone, atol=1e-5)

    def test_stereoify_handles_one_by_n(self):
        # The exact shape produced by soundfile.read(always_2d=True) on mono, then .T.
        one_by_n = self.tone.reshape(1, -1)
        out = mixer.stereoify(one_by_n)
        self.assertEqual(out.shape, (2, self.tone.shape[0]))
        np.testing.assert_array_equal(out[0], out[1])

    def test_stereoify_handles_n_by_one(self):
        n_by_one = self.tone.reshape(-1, 1)
        out = mixer.stereoify(n_by_one)
        self.assertEqual(out.shape, (2, self.tone.shape[0]))
        np.testing.assert_array_equal(out[0], out[1])


@unittest.skipUnless(HAVE_SOUNDFILE_FOR_TESTS, "soundfile not installed")
class DeesserRoleTests(unittest.TestCase):
    """De-essing is driven by chain config, not by hard-coded role names."""

    def test_non_vocal_role_can_be_de_essed(self):
        cfg = base_config()
        cfg["stem_detection"]["harmony"] = ["harmony"]
        cfg["chains"]["harmony"] = {
            "hpf_hz": 80.0,
            "stereo_width": 1.0,
            "deesser": {
                "mode": "builtin",
                "frequency_hz": 6200.0,
                "threshold_db": -28.0,
                "ratio": 4.0,
                "amount": 0.5,
                "attack_ms": 2.0,
                "release_ms": 90.0,
            },
            "compressor": {
                "threshold_db": -18.0,
                "ratio": 2.0,
                "attack_ms": 10.0,
                "release_ms": 80.0,
            },
            "eq_peaks": [],
            "parallel_comp_blend": 0.0,
        }

        sr = cfg["sample_rate"]
        tone = (0.3 * np.sin(2 * math.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song_harmony.wav"
            sf.write(str(path), tone, sr, subtype="FLOAT")
            stem = mixer.Stem(path, sr, cfg)
            stem.process(chunk_samples=4096)

        self.assertEqual(stem.role, "harmony")
        self.assertEqual(stem.stats["deesser_mode"], "builtin")

    def test_external_deesser_without_path_falls_back_to_builtin(self):
        cfg = base_config()
        cfg["chains"]["vocals"]["deesser"] = {
            "mode": "external",
            # Notably: no external_deesser_vst3 key.
            "frequency_hz": 6000.0,
            "ratio": 4.0,
            "amount": 0.5,
        }
        sr = cfg["sample_rate"]
        tone = (0.3 * np.sin(2 * math.pi * 440 * np.arange(sr) / sr)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song_vocal.wav"
            sf.write(str(path), tone, sr, subtype="FLOAT")
            stem = mixer.Stem(path, sr, cfg)
            mode, plugin, plug_path = stem._resolve_deesser()

        self.assertEqual(stem.role, "vocals")
        self.assertEqual(mode, "builtin")
        self.assertIsNone(plugin)
        self.assertIsNone(plug_path)
        self.assertTrue(
            any("no external_deesser_vst3 path configured" in d for d in stem.diagnostics),
            f"expected 'no path configured' diagnostic, got: {stem.diagnostics}",
        )

    def test_role_with_no_deesser_config_stays_off(self):
        cfg = base_config()
        sr = cfg["sample_rate"]
        tone = (0.3 * np.sin(2 * math.pi * 60 * np.arange(sr) / sr)).astype(np.float32)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "song_drum.wav"
            sf.write(str(path), tone, sr, subtype="FLOAT")
            stem = mixer.Stem(path, sr, cfg)
            stem.process(chunk_samples=4096)

        self.assertEqual(stem.role, "drums")
        self.assertEqual(stem.stats["deesser_mode"], "off")


class ReferenceCacheTests(unittest.TestCase):
    """The reference audio file is read at most once across the master chain."""

    def test_reference_read_once_when_both_consumers_active(self):
        cfg = base_config()
        # Activate both reference consumers: tonal-balance report + LUFS targeting.
        cfg["master"]["reference_match"]["tonal_balance_mode"] = "report"

        sr = cfg["sample_rate"]
        t = np.arange(sr) / sr
        library = {
            "song_vocal.wav": np.vstack([0.2 * np.sin(2 * math.pi * 440 * t)] * 2).astype(np.float32),
            "song_drum.wav": np.vstack([0.4 * np.sin(2 * math.pi * 60 * t)] * 2).astype(np.float32),
            "song_bass.wav": np.vstack([0.35 * np.sin(2 * math.pi * 80 * t)] * 2).astype(np.float32),
            "ref.wav": np.vstack([0.3 * np.sin(2 * math.pi * 200 * t)] * 2).astype(np.float32),
        }

        ref_reads = 0

        def counting_read(path, target_sr):
            nonlocal ref_reads
            if Path(path).name == "ref.wav":
                ref_reads += 1
            return library[Path(path).name], target_sr

        with mock.patch.object(mixer, "read_audio_file", side_effect=counting_read), \
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
                reference=Path("ref.wav"),
            )
            instance.process_stems()
            instance.build_buses()
            instance.render_mix()
            instance.master_chain()

        # Both `_apply_reference_tonal_balance` (mode=report) and
        # `_apply_loudness_and_reference` (LUFS) hit the reference path —
        # without caching this would be 2.
        self.assertEqual(ref_reads, 1)


class PanLawTests(unittest.TestCase):
    """Equal-power pan: cos/sin gains, constant L²+R² across pan positions."""

    def setUp(self):
        sr = 48000
        t = np.arange(sr // 4) / sr
        tone = (0.5 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        self.mono = np.vstack([tone, tone])  # mono content duplicated to L/R

    def test_center_uses_root_two_over_two_gains(self):
        out = mixer.apply_pan(self.mono, 0.0)
        expected = math.sqrt(0.5)
        np.testing.assert_allclose(out[0], self.mono[0] * expected, atol=1e-6)
        np.testing.assert_allclose(out[1], self.mono[1] * expected, atol=1e-6)
        # The new behavior is intentional: pan=0 is no longer a no-op.
        self.assertFalse(np.array_equal(out, self.mono))

    def test_summed_power_constant_across_pan_positions(self):
        # cos²θ + sin²θ = 1 → for mono-duplicated input, ||L||² + ||R||² is
        # constant for any pan. This is the defining equal-power property.
        energies = [
            float(np.sum(mixer.apply_pan(self.mono, p) ** 2))
            for p in (-1.0, -0.5, -0.15, 0.0, 0.15, 0.5, 1.0)
        ]
        spread = max(energies) - min(energies)
        self.assertLess(spread, 1e-6 * energies[0])

    def test_hard_pans_silence_opposite_channel(self):
        left = mixer.apply_pan(self.mono, -1.0)
        right = mixer.apply_pan(self.mono, 1.0)
        self.assertLess(float(np.max(np.abs(left[1]))), 1e-6)
        self.assertLess(float(np.max(np.abs(right[0]))), 1e-6)
        # The driven channel keeps full original amplitude.
        np.testing.assert_allclose(left[0], self.mono[0], atol=1e-6)
        np.testing.assert_allclose(right[1], self.mono[1], atol=1e-6)

    def test_non_stereo_input_passes_through(self):
        mono_1d = self.mono[0]
        out = mixer.apply_pan(mono_1d, 0.5)
        np.testing.assert_array_equal(out, mono_1d)


class RoleDetectionTests(unittest.TestCase):
    """Token-aware role detection: substrings inside other words must not match."""

    def setUp(self):
        # Mirrors config.yaml's stem_detection block.
        self.det = {
            "vocals": ["vox", "vocal", "lead", "singer"],
            "drums": ["drum", "drums", "kit", "percussion"],
            "bass": ["bass", "sub"],
            "guitar": ["gtr", "guitar"],
            "keys": ["keys", "piano", "synth", "pad", "rhodes", "organ"],
        }

    def _detect(self, name: str) -> str:
        return mixer.Stem._detect_role(Path(name), self.det)

    def test_classic_filenames_still_detect(self):
        self.assertEqual(self._detect("song_vocals_lead.wav"), "vocals")
        self.assertEqual(self._detect("song_drums_kick.wav"), "drums")
        self.assertEqual(self._detect("song_bass.wav"), "bass")
        self.assertEqual(self._detect("song_gtr_left.wav"), "guitar")
        self.assertEqual(self._detect("song_piano.wav"), "keys")

    def test_substring_collisions_no_longer_misclassify(self):
        # Pre-fix: "sub" matched submix → bass; "pad" matched padded_drums → keys.
        self.assertEqual(self._detect("submix.wav"), "default")
        self.assertEqual(self._detect("padded_drums.wav"), "drums")
        # subtle_synth still legitimately matches keys (via "synth"), just no longer via "sub".
        self.assertEqual(self._detect("subtle_synth.wav"), "keys")

    def test_plural_singular_tolerance(self):
        # Detection key "vocal" matches a "vocals.wav" filename even though the
        # token is plural; same for "drum" against "drums.wav".
        self.assertEqual(self._detect("vocals.wav"), "vocals")
        self.assertEqual(self._detect("drums.wav"), "drums")
        # Reverse direction: "keys" key matches a "key.wav" filename.
        self.assertEqual(self._detect("key.wav"), "keys")

    def test_camelcase_filenames_split(self):
        self.assertEqual(self._detect("SongVocals.wav"), "vocals")
        self.assertEqual(self._detect("KickDrum.wav"), "drums")

    def test_letter_digit_boundary_split(self):
        self.assertEqual(self._detect("Vocal01.wav"), "vocals")
        self.assertEqual(self._detect("drum_take_2.wav"), "drums")

    def test_unknown_filename_returns_default(self):
        self.assertEqual(self._detect("foo_bar.wav"), "default")
        self.assertEqual(self._detect("submix.wav"), "default")


if __name__ == "__main__":
    unittest.main()
