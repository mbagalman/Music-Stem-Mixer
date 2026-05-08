#!/usr/bin/env python3
"""
stem_mixer_oop.py
=================
Automatic stem mixer/master pipeline driven by YAML config.

This build focuses on better default mixing behavior for Suno-style stems:
- compatibility-aware config loading with validation and deprecation warnings
- per-role buses instead of per-file normalization
- built-in de-esser fallback with optional external plugin support
- diagnostics and analyze-only mode
- master limiting that enforces a real ceiling
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict

try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:  # pragma: no cover - runtime dependency guard
    np = None  # type: ignore[assignment]
    HAVE_NUMPY = False

try:
    import soundfile as sf
    HAVE_SOUNDFILE = True
except Exception:  # pragma: no cover - runtime dependency guard
    sf = None  # type: ignore[assignment]
    HAVE_SOUNDFILE = False

try:
    from pedalboard import (
        Compressor,
        HighpassFilter,
        HighShelfFilter,
        PeakFilter,
        Pedalboard,
        load_plugin,
    )
    HAVE_PEDALBOARD = True
except Exception:  # pragma: no cover - runtime dependency guard
    Compressor = HighpassFilter = HighShelfFilter = PeakFilter = Pedalboard = load_plugin = None  # type: ignore[assignment]
    HAVE_PEDALBOARD = False

try:
    import pyloudnorm as pyln
    HAVE_LOUDNORM = True
except Exception:
    pyln = None  # type: ignore[assignment]
    HAVE_LOUDNORM = False

try:
    import scipy.signal as signal
    HAVE_SCIPY = True
except Exception:
    signal = None  # type: ignore[assignment]
    HAVE_SCIPY = False

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:
    njit = None  # type: ignore[assignment]
    HAVE_NUMBA = False

try:
    import yaml
    HAVE_YAML = True
except Exception:
    yaml = None  # type: ignore[assignment]
    HAVE_YAML = False


# ---------- Typed config ----------
class EqPeak(TypedDict):
    freq: float
    q: float
    gain_db: float


class CompCfg(TypedDict, total=False):
    threshold_db: float
    ratio: float
    attack_ms: float
    release_ms: float
    makeup_db: float


class DeesserCfg(TypedDict, total=False):
    mode: str
    frequency_hz: float
    threshold_db: float
    ratio: float
    amount: float
    attack_ms: float
    release_ms: float
    external_deesser_vst3: str


class ChainCfg(TypedDict, total=False):
    hpf_hz: float
    stereo_width: float
    compressor: CompCfg
    deesser: DeesserCfg
    external_deesser_vst3: str
    eq_peaks: List[EqPeak]
    parallel_comp_blend: float
    target_rms_dbfs: float


class BusCfg(TypedDict, total=False):
    target_rms_dbfs: float
    gain_db: float
    pan: float
    stereo_width: float
    mono_below_hz: float


class SidechainCfg(TypedDict, total=False):
    enabled: bool
    trigger: str
    targets: List[str]
    amount: float
    attack_ms: float
    release_ms: float
    mode: str
    low_band_hz: float


class MultibandCfg(TypedDict, total=False):
    enabled: bool
    bands: List[float]
    ratios: List[float]
    threshold_db: float
    attack_ms: float
    release_ms: float


class LimiterCfg(TypedDict, total=False):
    ceiling_dbfs: float
    release_ms: float


class ReferenceMatchCfg(TypedDict, total=False):
    tonal_balance_mode: str
    amount: float
    max_tilt_db_per_octave: float


class AlignmentCfg(TypedDict, total=False):
    enabled: bool
    threshold_dbfs: float
    min_offset_ms: float


class MasterCfg(TypedDict, total=False):
    hpf_hz: float
    high_shelf: Dict[str, float]
    compressor: CompCfg
    limiter_ceil_dbfs: float
    limiter: LimiterCfg
    target_lufs: float
    multiband: MultibandCfg
    reference_match: ReferenceMatchCfg


class FullConfig(TypedDict, total=False):
    sample_rate: int
    stem_detection: Dict[str, List[str]]
    chains: Dict[str, ChainCfg]
    buses: Dict[str, BusCfg]
    sidechain: SidechainCfg
    master: MasterCfg
    alignment: AlignmentCfg


# ---------- Defaults ----------
DEFAULT_LIMITER: LimiterCfg = {"ceiling_dbfs": -1.0, "release_ms": 120.0}
DEFAULT_DEESSER: DeesserCfg = {
    "mode": "off",
    "frequency_hz": 6000.0,
    "threshold_db": -28.0,
    "ratio": 4.0,
    "amount": 0.6,
    "attack_ms": 2.0,
    "release_ms": 90.0,
}
DEFAULT_BUS: BusCfg = {
    "target_rms_dbfs": -20.0,
    "gain_db": 0.0,
    "pan": 0.0,
    "stereo_width": 1.0,
}
DEFAULT_ALIGNMENT: AlignmentCfg = {
    "enabled": False,
    "threshold_dbfs": -45.0,
    "min_offset_ms": 25.0,
}
DEFAULT_REFERENCE_MATCH: ReferenceMatchCfg = {
    "tonal_balance_mode": "off",
    "amount": 0.5,
    "max_tilt_db_per_octave": 3.0,
}


# ---------- Helpers ----------
def require_runtime_dependencies() -> None:
    missing = []
    if not HAVE_NUMPY:
        missing.append("numpy")
    if not HAVE_SOUNDFILE:
        missing.append("soundfile")
    if not HAVE_PEDALBOARD:
        missing.append("pedalboard")
    if not HAVE_YAML:
        missing.append("pyyaml")
    if missing:
        raise SystemExit(f"Missing required dependencies: {', '.join(missing)}. Run: pip install -r requirements.txt")


def db_to_amp(db: float) -> float:
    return 10.0 ** (db / 20.0)


def amp_to_db(amp: float) -> float:
    return 20.0 * math.log10(max(float(amp), 1e-12))


def stereoify(y: np.ndarray) -> np.ndarray:
    if y.ndim == 1:
        return np.vstack([y, y])
    if y.ndim != 2:
        raise ValueError(f"Expected 1D mono or 2D stereo audio, got shape {y.shape}")
    if y.shape[0] == 2:
        return y
    if y.shape[1] == 2:
        return y.T
    # soundfile.read(always_2d=True) returns (frames, 1) for mono → (1, frames) after .T.
    if y.shape[0] == 1:
        return np.vstack([y[0], y[0]])
    if y.shape[1] == 1:
        return np.vstack([y[:, 0], y[:, 0]])
    raise ValueError(f"Unsupported channel layout {y.shape}; expected mono or stereo input")


def apply_gain_db(y: np.ndarray, gain_db: float) -> np.ndarray:
    return y * db_to_amp(gain_db)


def calc_rms_dbfs(y: np.ndarray) -> float:
    eps = 1e-12
    rms = float(np.sqrt(np.mean(np.square(y)) + eps))
    return amp_to_db(rms)


def calc_peak_dbfs(y: np.ndarray) -> float:
    return amp_to_db(float(np.max(np.abs(y))))


def safe_peak_normalize(y: np.ndarray, peak_dbfs: float = -1.0) -> np.ndarray:
    peak = float(np.max(np.abs(y)))
    if peak <= 0:
        return y
    target_amp = db_to_amp(peak_dbfs)
    return y * (target_amp / peak) if peak > target_amp else y


def apply_stereo_width(y: np.ndarray, width: float, normalize_peaks: bool = True) -> np.ndarray:
    if y.shape[0] != 2 or abs(width - 1.0) < 1e-6:
        return y
    left, right = y[0], y[1]
    mid = 0.5 * (left + right)
    side = 0.5 * (left - right) * float(width)
    out = np.vstack([mid + side, mid - side])
    if normalize_peaks:
        peak_in = float(np.max(np.abs(y))) + 1e-12
        peak_out = float(np.max(np.abs(out))) + 1e-12
        if peak_out > peak_in:
            out *= peak_in / peak_out
    return out


def apply_pan(y: np.ndarray, pan: float) -> np.ndarray:
    # Equal-power (constant-power) pan: θ ∈ [0, π/2] across pan ∈ [-1, +1].
    # At pan=0 both gains drop to √0.5 ≈ 0.707, which is the intended center —
    # not a no-op. Bus RMS gain-staging (applied after pan) absorbs the level shift.
    if y.shape[0] != 2:
        return y
    pan = float(max(-1.0, min(1.0, pan)))
    theta = (pan + 1.0) * (math.pi / 4.0)
    left_gain = math.cos(theta)
    right_gain = math.sin(theta)
    return np.vstack([y[0] * left_gain, y[1] * right_gain])


def leading_silence_samples(y: np.ndarray, threshold_dbfs: float) -> int:
    if y.size == 0:
        return 0
    threshold_amp = db_to_amp(threshold_dbfs)
    mono = np.max(np.abs(y), axis=0)
    above = np.flatnonzero(mono > threshold_amp)
    return int(above[0]) if len(above) else int(y.shape[1])


def warn_deprecated(message: str) -> None:
    warnings.warn(message, UserWarning, stacklevel=2)


def _validate_positive(value: float, label: str) -> None:
    if float(value) <= 0.0:
        raise ValueError(f"{label} must be > 0, got {value}")


def _validate_ratio(value: float, label: str) -> None:
    if float(value) < 1.0:
        raise ValueError(f"{label} must be >= 1, got {value}")


def _validate_range(value: float, label: str, lo: float, hi: float) -> None:
    if not (lo <= float(value) <= hi):
        raise ValueError(f"{label} out of range ({lo}..{hi}), got {value}")


def resample_audio(y: np.ndarray, src_sr: int, target_sr: int) -> np.ndarray:
    if src_sr == target_sr:
        return y
    if HAVE_SCIPY:
        gcd = math.gcd(int(src_sr), int(target_sr))
        up = int(target_sr // gcd)
        down = int(src_sr // gcd)
        return signal.resample_poly(y, up, down, axis=1).astype(np.float32)
    raise RuntimeError("Sample rate conversion requires scipy")


def envelope_follower(
    x: np.ndarray,
    sr: int,
    attack_ms: float,
    release_ms: float,
    normalize: bool = True,
) -> np.ndarray:
    x = np.abs(x).astype(np.float32, copy=False)
    a_attack = math.exp(-1.0 / (sr * (attack_ms / 1000.0)))
    a_release = math.exp(-1.0 / (sr * (release_ms / 1000.0)))
    if HAVE_NUMBA:
        env = _env_loop(x, a_attack, a_release)
    else:
        env = np.empty_like(x, dtype=np.float32)
        prev = 0.0
        for i in range(len(x)):
            coeff = a_attack if x[i] > prev else a_release
            prev = (1 - coeff) * x[i] + coeff * prev
            env[i] = prev
    if normalize:
        p99 = float(np.percentile(env, 99)) + 1e-9
        env = np.clip(env / p99, 0.0, 1.0)
    return env


if HAVE_NUMBA:
    @njit(cache=True)
    def _env_loop(x: np.ndarray, a_attack: float, a_release: float) -> np.ndarray:  # pragma: no cover - numba path
        n = len(x)
        env = np.zeros(n, dtype=np.float32)
        prev = 0.0
        for i in range(n):
            coeff = a_attack if x[i] > prev else a_release
            prev = (1 - coeff) * x[i] + coeff * prev
            env[i] = prev
        return env


def lr4_sos(cut_hz: float, sr: int, btype: str) -> Optional[np.ndarray]:
    if not HAVE_SCIPY:
        return None
    sos2 = signal.butter(2, cut_hz, btype=btype, fs=sr, output="sos")
    return np.vstack([sos2, sos2])


def lr4_lowpass(x: np.ndarray, cut_hz: float, sr: int) -> np.ndarray:
    sos = lr4_sos(cut_hz, sr, "lowpass")
    return signal.sosfiltfilt(sos, x) if sos is not None else x


def lr4_highpass(x: np.ndarray, cut_hz: float, sr: int) -> np.ndarray:
    sos = lr4_sos(cut_hz, sr, "highpass")
    return signal.sosfiltfilt(sos, x) if sos is not None else x


def split_low_high(y: np.ndarray, cut_hz: float, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    if HAVE_SCIPY:
        low = np.vstack([lr4_lowpass(y[0], cut_hz, sr), lr4_lowpass(y[1], cut_hz, sr)])
        high = y - low
        return low, high
    low = np.vstack([np.copy(y[0]), np.copy(y[1])])
    high = np.zeros_like(y)
    return low, high


def apply_mono_below(y: np.ndarray, cut_hz: float, sr: int) -> np.ndarray:
    if y.shape[0] != 2 or cut_hz <= 0:
        return y
    low, high = split_low_high(y, cut_hz, sr)
    mono_low = np.mean(low, axis=0)
    return np.vstack([mono_low, mono_low]) + high


def apply_builtin_deesser(y: np.ndarray, sr: int, cfg: DeesserCfg) -> np.ndarray:
    frequency_hz = float(cfg.get("frequency_hz", DEFAULT_DEESSER["frequency_hz"]))
    threshold_db = float(cfg.get("threshold_db", DEFAULT_DEESSER["threshold_db"]))
    ratio = float(cfg.get("ratio", DEFAULT_DEESSER["ratio"]))
    amount = float(cfg.get("amount", DEFAULT_DEESSER["amount"]))
    attack_ms = float(cfg.get("attack_ms", DEFAULT_DEESSER["attack_ms"]))
    release_ms = float(cfg.get("release_ms", DEFAULT_DEESSER["release_ms"]))

    low, high = split_low_high(y, frequency_hz, sr)
    if np.max(np.abs(high)) <= 0:
        return y

    sidechain = np.mean(np.abs(high), axis=0)
    env = envelope_follower(sidechain, sr, attack_ms, release_ms, normalize=False)
    threshold_amp = db_to_amp(threshold_db)
    over = np.maximum(env / max(threshold_amp, 1e-9), 1.0)
    gain = np.minimum(1.0, 1.0 / np.power(over, (ratio - 1.0) / max(ratio, 1.0)))
    gain = 1.0 - amount * (1.0 - gain)
    return low + high * gain[np.newaxis, :]


def _limiter_compute_gains(peaks: np.ndarray, ceiling_amp: float, release_coeff: float) -> np.ndarray:
    n = peaks.shape[0]
    gains = np.empty(n, dtype=np.float32)
    current_gain = 1.0
    for i in range(n):
        peak = peaks[i]
        if peak < 1e-12:
            peak = 1e-12
        required_gain = ceiling_amp / peak
        if required_gain > 1.0:
            required_gain = 1.0
        if required_gain < current_gain:
            current_gain = required_gain
        else:
            current_gain = release_coeff * current_gain + (1.0 - release_coeff)
            if current_gain > 1.0:
                current_gain = 1.0
        gains[i] = current_gain
    return gains


if HAVE_NUMBA:
    _limiter_compute_gains = njit(cache=True)(_limiter_compute_gains)  # type: ignore[assignment]


def apply_peak_limiter(y: np.ndarray, sr: int, ceiling_dbfs: float, release_ms: float) -> np.ndarray:
    ceiling_amp = db_to_amp(ceiling_dbfs)
    release_coeff = math.exp(-1.0 / (sr * (release_ms / 1000.0)))
    peaks = np.max(np.abs(y), axis=0).astype(np.float32, copy=False)
    gains = _limiter_compute_gains(peaks, float(ceiling_amp), float(release_coeff))
    return safe_peak_normalize(y * gains[np.newaxis, :], peak_dbfs=ceiling_dbfs)


def compute_spectral_tilt(y: np.ndarray, sr: int) -> float:
    mono = np.mean(y, axis=0)
    if mono.size < 64:
        return 0.0
    spec = np.fft.rfft(mono)
    freqs = np.fft.rfftfreq(mono.size, d=1.0 / sr)
    mask = (freqs >= 40.0) & (freqs <= min(16000.0, sr / 2.0 - 1.0))
    if not np.any(mask):
        return 0.0
    mags_db = 20.0 * np.log10(np.abs(spec[mask]) + 1e-9)
    octaves = np.log2(freqs[mask] / 1000.0)
    slope, _ = np.polyfit(octaves, mags_db, 1)
    return float(slope)


def apply_spectral_tilt(y: np.ndarray, sr: int, tilt_db_per_octave: float) -> np.ndarray:
    if abs(tilt_db_per_octave) < 1e-6:
        return y
    out = np.empty_like(y)
    for ch in range(y.shape[0]):
        spec = np.fft.rfft(y[ch])
        freqs = np.fft.rfftfreq(y[ch].size, d=1.0 / sr)
        safe_freqs = np.maximum(freqs, 20.0)
        octaves = np.log2(safe_freqs / 1000.0)
        tilt = np.power(10.0, (tilt_db_per_octave * octaves) / 20.0)
        tilt[0] = 1.0
        out[ch] = np.fft.irfft(spec * tilt, n=y[ch].size).astype(np.float32)
    return safe_peak_normalize(out, calc_peak_dbfs(y))


def run_board(audio: np.ndarray, board: Iterable[Any], sr: int, chunk_samples: int) -> np.ndarray:
    board_list = list(board)
    if not board_list:
        return audio
    if not HAVE_PEDALBOARD:
        raise RuntimeError("Pedalboard is required for processing effects")
    pedalboard = Pedalboard(board_list)
    out = np.empty_like(audio)
    n = audio.shape[1]
    for start in range(0, n, chunk_samples):
        end = min(start + chunk_samples, n)
        out[:, start:end] = pedalboard.process(audio[:, start:end].T, sr).T
    return out


def read_audio_file(path: Path, target_sr: int) -> Tuple[np.ndarray, int]:
    if not HAVE_SOUNDFILE:
        raise RuntimeError("soundfile is required to read audio")
    audio, src_sr = sf.read(str(path), always_2d=True)
    audio = audio.T.astype(np.float32)
    audio = stereoify(audio)
    if src_sr != target_sr:
        audio = resample_audio(audio, src_sr, target_sr)
        src_sr = target_sr
    return np.ascontiguousarray(audio), src_sr


def normalize_config(cfg: Dict[str, Any]) -> FullConfig:
    if not isinstance(cfg, dict):
        raise ValueError("Top-level config must be a mapping")
    cfg.setdefault("stem_detection", {})
    cfg.setdefault("chains", {})
    cfg.setdefault("buses", {})
    cfg.setdefault("sidechain", {})
    cfg.setdefault("alignment", {})
    cfg.setdefault("master", {})

    chains = cfg["chains"]
    if "default" not in chains:
        raise ValueError("Config must define chains.default")

    cfg["alignment"] = {**DEFAULT_ALIGNMENT, **cfg.get("alignment", {})}

    master = cfg["master"]
    if "limiter" not in master:
        master["limiter"] = dict(DEFAULT_LIMITER)
    else:
        master["limiter"] = {**DEFAULT_LIMITER, **master["limiter"]}
    if "limiter_ceil_dbfs" in master:
        warn_deprecated("master.limiter_ceil_dbfs is deprecated; use master.limiter.ceiling_dbfs")
        master["limiter"]["ceiling_dbfs"] = float(master["limiter_ceil_dbfs"])
    master.setdefault("reference_match", {})
    master["reference_match"] = {**DEFAULT_REFERENCE_MATCH, **master["reference_match"]}

    buses = cfg["buses"]
    buses.setdefault("default", dict(DEFAULT_BUS))
    buses["default"] = {**DEFAULT_BUS, **buses["default"]}

    for role, chain in list(chains.items()):
        if "external_deesser_vst3" in chain:
            deesser = dict(DEFAULT_DEESSER)
            deesser.update(chain.get("deesser", {}))
            deesser["mode"] = "external"
            deesser["external_deesser_vst3"] = chain["external_deesser_vst3"]
            chain["deesser"] = deesser
            warn_deprecated(
                f"chains.{role}.external_deesser_vst3 is deprecated; move it under chains.{role}.deesser.external_deesser_vst3"
            )
        elif "deesser" in chain:
            chain["deesser"] = {**DEFAULT_DEESSER, **chain["deesser"]}

        bus_cfg = dict(DEFAULT_BUS)
        bus_cfg.update(buses.get(role, {}))
        if "target_rms_dbfs" in chain and "target_rms_dbfs" not in buses.get(role, {}):
            bus_cfg["target_rms_dbfs"] = float(chain["target_rms_dbfs"])
            warn_deprecated(
                f"chains.{role}.target_rms_dbfs now controls buses.{role}.target_rms_dbfs via compatibility shim"
            )
        buses[role] = bus_cfg

    return cfg  # type: ignore[return-value]


def validate_config(cfg: FullConfig) -> None:
    if "sample_rate" not in cfg:
        raise ValueError("sample_rate is required")
    _validate_positive(float(cfg["sample_rate"]), "sample_rate")

    if "chains" not in cfg or "default" not in cfg["chains"]:
        raise ValueError("chains.default is required")
    if "master" not in cfg:
        raise ValueError("master is required")

    for role, chain in cfg["chains"].items():
        _validate_range(float(chain.get("stereo_width", 1.0)), f"chains.{role}.stereo_width", 0.0, 2.0)
        _validate_range(float(chain.get("parallel_comp_blend", 0.0)), f"chains.{role}.parallel_comp_blend", 0.0, 1.0)
        if "compressor" in chain:
            comp = chain["compressor"]
            _validate_ratio(float(comp.get("ratio", 1.0)), f"chains.{role}.compressor.ratio")
            _validate_positive(float(comp.get("attack_ms", 1.0)), f"chains.{role}.compressor.attack_ms")
            _validate_positive(float(comp.get("release_ms", 1.0)), f"chains.{role}.compressor.release_ms")
        if "deesser" in chain:
            deesser = chain["deesser"]
            if deesser.get("mode", "off") not in {"off", "builtin", "external"}:
                raise ValueError(f"chains.{role}.deesser.mode must be off|builtin|external")
            _validate_range(float(deesser.get("amount", DEFAULT_DEESSER["amount"])), f"chains.{role}.deesser.amount", 0.0, 1.0)
            _validate_positive(float(deesser.get("frequency_hz", DEFAULT_DEESSER["frequency_hz"])), f"chains.{role}.deesser.frequency_hz")
            _validate_ratio(float(deesser.get("ratio", DEFAULT_DEESSER["ratio"])), f"chains.{role}.deesser.ratio")

    for role, bus in cfg.get("buses", {}).items():
        _validate_range(float(bus.get("pan", 0.0)), f"buses.{role}.pan", -1.0, 1.0)
        _validate_range(float(bus.get("stereo_width", 1.0)), f"buses.{role}.stereo_width", 0.0, 2.0)
        _validate_range(float(bus.get("target_rms_dbfs", -20.0)), f"buses.{role}.target_rms_dbfs", -60.0, 0.0)
        mono_below_hz = float(bus.get("mono_below_hz", 0.0))
        if mono_below_hz:
            _validate_positive(mono_below_hz, f"buses.{role}.mono_below_hz")

    sc = cfg.get("sidechain", {})
    if sc.get("enabled"):
        _validate_range(float(sc.get("amount", 0.3)), "sidechain.amount", 0.0, 1.0)
        _validate_positive(float(sc.get("attack_ms", 5.0)), "sidechain.attack_ms")
        _validate_positive(float(sc.get("release_ms", 120.0)), "sidechain.release_ms")
        if sc.get("mode", "broadband") not in {"broadband", "low_band"}:
            raise ValueError("sidechain.mode must be broadband|low_band")
        if sc.get("mode", "broadband") == "low_band":
            _validate_positive(float(sc.get("low_band_hz", 150.0)), "sidechain.low_band_hz")
        if not sc.get("targets"):
            raise ValueError("sidechain.targets must contain at least one role when enabled")

    alignment = cfg.get("alignment", DEFAULT_ALIGNMENT)
    _validate_positive(float(alignment.get("min_offset_ms", DEFAULT_ALIGNMENT["min_offset_ms"])), "alignment.min_offset_ms")

    master = cfg["master"]
    limiter = master.get("limiter", DEFAULT_LIMITER)
    _validate_range(float(limiter.get("ceiling_dbfs", DEFAULT_LIMITER["ceiling_dbfs"])), "master.limiter.ceiling_dbfs", -20.0, 0.0)
    _validate_positive(float(limiter.get("release_ms", DEFAULT_LIMITER["release_ms"])), "master.limiter.release_ms")
    if "compressor" in master:
        comp = master["compressor"]
        _validate_ratio(float(comp.get("ratio", 1.0)), "master.compressor.ratio")
        _validate_positive(float(comp.get("attack_ms", 1.0)), "master.compressor.attack_ms")
        _validate_positive(float(comp.get("release_ms", 1.0)), "master.compressor.release_ms")

    mb = master.get("multiband", {})
    if mb.get("enabled"):
        bands = [float(b) for b in mb.get("bands", [])]
        ratios = [float(r) for r in mb.get("ratios", [])]
        if not bands:
            raise ValueError("master.multiband.bands must not be empty when multiband is enabled")
        if bands != sorted(bands):
            raise ValueError("master.multiband.bands must be strictly ascending")
        if any(b <= 0 for b in bands):
            raise ValueError("master.multiband.bands must all be > 0")
        if len(ratios) != len(bands) + 1:
            raise ValueError("master.multiband.ratios must have length len(bands)+1")
        if any(r < 1.0 for r in ratios):
            raise ValueError("master.multiband.ratios must be >= 1")
        _validate_positive(float(mb.get("attack_ms", 10.0)), "master.multiband.attack_ms")
        _validate_positive(float(mb.get("release_ms", 120.0)), "master.multiband.release_ms")

    ref_match = master.get("reference_match", DEFAULT_REFERENCE_MATCH)
    if ref_match.get("tonal_balance_mode", "off") not in {"off", "report", "apply"}:
        raise ValueError("master.reference_match.tonal_balance_mode must be off|report|apply")
    _validate_range(float(ref_match.get("amount", 0.5)), "master.reference_match.amount", 0.0, 1.0)
    _validate_positive(
        float(ref_match.get("max_tilt_db_per_octave", DEFAULT_REFERENCE_MATCH["max_tilt_db_per_octave"])),
        "master.reference_match.max_tilt_db_per_octave",
    )


def load_config(path: Optional[Path]) -> FullConfig:
    if path is None:
        raise SystemExit("--config is required (YAML)")
    if not HAVE_YAML:
        raise SystemExit("PyYAML not installed. Run: pip install -r requirements.txt")
    cfg = yaml.safe_load(path.read_text()) or {}
    normalized = normalize_config(cfg)
    validate_config(normalized)
    return normalized


# Splits a file stem into tokens on non-alphanumeric runs, CamelCase boundaries, and letter↔digit boundaries.
_TOKEN_SPLIT = re.compile(r"[^a-zA-Z0-9]+|(?<=[a-z])(?=[A-Z])|(?<=[a-zA-Z])(?=\d)|(?<=\d)(?=[a-zA-Z])")


class Stem:
    def __init__(self, path: Path, sr: int, cfg: FullConfig):
        self.path = path
        self.sr = sr
        self.role = self._detect_role(path, cfg["stem_detection"])
        self.cfg = cfg["chains"].get(self.role, cfg["chains"]["default"])
        self.audio, _ = read_audio_file(path, sr)
        self.input_stats: Dict[str, Any] = {}
        self.stats: Dict[str, Any] = {}
        self.diagnostics: List[str] = []
        self.alignment_trimmed_samples = 0
        self._analyze_input(cfg.get("alignment", DEFAULT_ALIGNMENT))

    @staticmethod
    def _detect_role(path: Path, det: Dict[str, List[str]]) -> str:
        tokens = {t.lower() for t in _TOKEN_SPLIT.split(path.stem) if t}
        for role, keys in det.items():
            for key in keys:
                key_lower = key.lower()
                if key_lower in tokens:
                    return role
                # Tolerate simple English plural mismatches between key and token.
                if key_lower + "s" in tokens:
                    return role
                if key_lower.endswith("s") and key_lower[:-1] in tokens:
                    return role
        return "default"

    def _analyze_input(self, alignment_cfg: AlignmentCfg) -> None:
        threshold_dbfs = float(alignment_cfg.get("threshold_dbfs", DEFAULT_ALIGNMENT["threshold_dbfs"]))
        leading = leading_silence_samples(self.audio, threshold_dbfs)
        peak = float(np.max(np.abs(self.audio)))
        rms_dbfs = calc_rms_dbfs(self.audio)
        self.input_stats = {
            "input_peak_dbfs": calc_peak_dbfs(self.audio),
            "input_rms_dbfs": rms_dbfs,
            "leading_silence_ms": (leading / self.sr) * 1000.0,
            "clipped": peak >= 0.999,
            "silent": rms_dbfs <= -55.0,
            "sample_count": int(self.audio.shape[1]),
        }
        if self.input_stats["clipped"]:
            self.diagnostics.append(f"{self.path.name}: input appears clipped")
        if self.input_stats["silent"]:
            self.diagnostics.append(f"{self.path.name}: input is nearly silent")
        if rms_dbfs > -8.0:
            self.diagnostics.append(f"{self.path.name}: unusually hot input ({rms_dbfs:.1f} dBFS RMS)")
        if rms_dbfs < -36.0:
            self.diagnostics.append(f"{self.path.name}: unusually quiet input ({rms_dbfs:.1f} dBFS RMS)")

    def trim_leading_samples(self, samples: int) -> None:
        if samples <= 0:
            return
        trim = min(samples, self.audio.shape[1])
        self.audio = self.audio[:, trim:]
        self.alignment_trimmed_samples += trim
        self.stats["alignment_trimmed_ms"] = (self.alignment_trimmed_samples / self.sr) * 1000.0

    def _resolve_deesser(self) -> Tuple[str, Optional[Any], Optional[str]]:
        deesser_cfg = dict(DEFAULT_DEESSER)
        deesser_cfg.update(self.cfg.get("deesser", {}))
        mode = deesser_cfg.get("mode", "off")
        if mode == "off":
            return "off", None, None
        if mode == "external":
            raw_path = deesser_cfg.get("external_deesser_vst3")
            plug_path = raw_path.strip() if isinstance(raw_path, str) else ""
            if not plug_path:
                self.diagnostics.append(
                    f"{self.path.name}: external de-esser requested but no external_deesser_vst3 path configured; falling back to builtin"
                )
                return "builtin", None, None
            try:
                plugin = load_plugin(plug_path) if HAVE_PEDALBOARD else None
                if plugin is not None:
                    return "external", plugin, plug_path
            except Exception as exc:
                self.diagnostics.append(f"{self.path.name}: external de-esser failed to load ({plug_path}): {exc}")
            self.diagnostics.append(f"{self.path.name}: external de-esser unavailable; falling back to builtin")
            return "builtin", None, None
        return "builtin", None, None

    def process(self, chunk_samples: int) -> None:
        effects = [HighpassFilter(float(self.cfg.get("hpf_hz", 50.0)))]
        for peak in self.cfg.get("eq_peaks", []):
            effects.append(
                PeakFilter(
                    cutoff_frequency_hz=float(peak["freq"]),
                    gain_db=float(peak["gain_db"]),
                    q=float(peak["q"]),
                )
            )
        self.audio = run_board(self.audio, effects, self.sr, chunk_samples)

        deesser_cfg = dict(DEFAULT_DEESSER)
        deesser_cfg.update(self.cfg.get("deesser", {}))
        deesser_mode, external_plugin, external_path = self._resolve_deesser()
        self.stats["deesser_mode"] = deesser_mode
        if deesser_mode == "external" and external_plugin is not None:
            self.audio = run_board(self.audio, [external_plugin], self.sr, chunk_samples)
            self.stats["deesser_plugin"] = external_path
        elif deesser_mode == "builtin":
            self.audio = apply_builtin_deesser(self.audio, self.sr, deesser_cfg)

        comp = self.cfg.get("compressor")
        if comp:
            comp_effect = Compressor(
                threshold_db=float(comp["threshold_db"]),
                ratio=float(comp["ratio"]),
                attack_ms=float(comp["attack_ms"]),
                release_ms=float(comp["release_ms"]),
            )
            self.audio = run_board(self.audio, [comp_effect], self.sr, chunk_samples)
            makeup_db = float(comp.get("makeup_db", 0.0))
            if abs(makeup_db) > 1e-6:
                self.audio = apply_gain_db(self.audio, makeup_db)

        blend = float(self.cfg.get("parallel_comp_blend", 0.0))
        if blend > 0.0:
            parallel_comp = Compressor(threshold_db=-30.0, ratio=10.0, attack_ms=0.5, release_ms=50.0)
            crushed = run_board(self.audio, [parallel_comp], self.sr, chunk_samples)
            self.audio = self.audio * (1.0 - blend) + crushed * blend

        width = float(self.cfg.get("stereo_width", 1.0))
        self.audio = apply_stereo_width(self.audio, width, normalize_peaks=True)
        self.stats["post_peak_dbfs"] = calc_peak_dbfs(self.audio)
        self.stats["post_rms_dbfs"] = calc_rms_dbfs(self.audio)

    def pad_to(self, n_samples: int) -> None:
        if self.audio.shape[1] < n_samples:
            pad = np.zeros((2, n_samples - self.audio.shape[1]), dtype=self.audio.dtype)
            self.audio = np.hstack([self.audio, pad])


class Bus:
    def __init__(self, role: str, stems: List[Stem], cfg: BusCfg, sr: int):
        self.role = role
        self.stems = stems
        self.cfg = {**DEFAULT_BUS, **cfg}
        self.sr = sr
        self.audio: np.ndarray = np.sum([stem.audio for stem in stems], axis=0) if stems else np.zeros((2, 0), dtype=np.float32)
        self.stats: Dict[str, Any] = {}

    def process(self) -> None:
        if self.audio.size == 0:
            return
        mono_below_hz = float(self.cfg.get("mono_below_hz", 0.0))
        if mono_below_hz > 0.0:
            self.audio = apply_mono_below(self.audio, mono_below_hz, self.sr)
        self.audio = apply_stereo_width(self.audio, float(self.cfg.get("stereo_width", 1.0)))
        self.audio = apply_pan(self.audio, float(self.cfg.get("pan", 0.0)))

        target_rms = float(self.cfg.get("target_rms_dbfs", DEFAULT_BUS["target_rms_dbfs"]))
        current_rms = calc_rms_dbfs(self.audio)
        self.audio = apply_gain_db(self.audio, target_rms - current_rms + float(self.cfg.get("gain_db", 0.0)))
        self.stats["post_rms_dbfs"] = calc_rms_dbfs(self.audio)
        self.stats["post_peak_dbfs"] = calc_peak_dbfs(self.audio)
        self.stats["stem_count"] = len(self.stems)


class Mixer:
    def __init__(
        self,
        files: List[Path],
        cfg: FullConfig,
        sr_cli: Optional[int],
        chunk_samples: int,
        reference: Optional[Path],
        analyze_only: bool = False,
    ):
        self.cfg = cfg
        self.sr = sr_cli or int(cfg["sample_rate"])
        self.chunk = int(chunk_samples)
        self.reference = reference
        self.analyze_only = analyze_only
        self.report: Dict[str, Any] = {"diagnostics": []}
        self.stems = [Stem(path, self.sr, cfg) for path in files]
        self.buses: Dict[str, Bus] = {}
        self.mix: Optional[np.ndarray] = None
        self._reference_audio: Optional[np.ndarray] = None
        self._collect_initial_diagnostics()
        self._apply_alignment_if_enabled()

    def _get_reference_audio(self) -> Optional[np.ndarray]:
        if self.reference is None:
            return None
        if self._reference_audio is None:
            self._reference_audio, _ = read_audio_file(self.reference, self.sr)
        return self._reference_audio

    def _collect_initial_diagnostics(self) -> None:
        diagnostics = self.report["diagnostics"]
        for stem in self.stems:
            diagnostics.extend(stem.diagnostics)
        role_counts: Dict[str, int] = {}
        for stem in self.stems:
            role_counts[stem.role] = role_counts.get(stem.role, 0) + 1
        self.report["role_counts"] = role_counts
        for role, count in sorted(role_counts.items()):
            if count > 1:
                diagnostics.append(f"Role '{role}' has {count} stems; bus normalization will control stack gain")

        leading = [float(stem.input_stats["leading_silence_ms"]) for stem in self.stems if not stem.input_stats.get("silent")]
        if leading:
            spread = max(leading) - min(leading)
            self.report["leading_silence_spread_ms"] = spread
            if spread >= float(self.cfg.get("alignment", DEFAULT_ALIGNMENT).get("min_offset_ms", DEFAULT_ALIGNMENT["min_offset_ms"])):
                diagnostics.append(f"Detected leading-silence spread of {spread:.1f} ms across stems")

    def _apply_alignment_if_enabled(self) -> None:
        alignment = self.cfg.get("alignment", DEFAULT_ALIGNMENT)
        if not alignment.get("enabled"):
            return
        threshold_dbfs = float(alignment.get("threshold_dbfs", DEFAULT_ALIGNMENT["threshold_dbfs"]))
        min_offset_ms = float(alignment.get("min_offset_ms", DEFAULT_ALIGNMENT["min_offset_ms"]))
        leading_samples = [
            leading_silence_samples(stem.audio, threshold_dbfs)
            for stem in self.stems
            if not stem.input_stats.get("silent")
        ]
        if not leading_samples:
            return
        anchor = min(leading_samples)
        min_shift_samples = int((min_offset_ms / 1000.0) * self.sr)
        for stem in self.stems:
            leading = leading_silence_samples(stem.audio, threshold_dbfs)
            shift = max(0, leading - anchor)
            if shift >= min_shift_samples:
                stem.trim_leading_samples(shift)
                self.report["diagnostics"].append(f"{stem.path.name}: trimmed {shift / self.sr * 1000.0:.1f} ms for alignment")

    def process_stems(self) -> None:
        for stem in self.stems:
            stem.process(self.chunk)
        max_len = max(stem.audio.shape[1] for stem in self.stems)
        for stem in self.stems:
            stem.pad_to(max_len)

    def build_buses(self) -> None:
        grouped: Dict[str, List[Stem]] = {}
        for stem in self.stems:
            grouped.setdefault(stem.role, []).append(stem)
        for role, stems in grouped.items():
            bus_cfg = self.cfg.get("buses", {}).get(role, self.cfg.get("buses", {}).get("default", DEFAULT_BUS))
            bus = Bus(role, stems, bus_cfg, self.sr)
            bus.process()
            self.buses[role] = bus

    def apply_sidechain(self) -> None:
        sc = self.cfg.get("sidechain", {})
        if not sc.get("enabled"):
            return
        trig_role = sc.get("trigger", "drums")
        targets = sc.get("targets", [])
        amount = float(sc.get("amount", 0.3))
        attack_ms = float(sc.get("attack_ms", 5.0))
        release_ms = float(sc.get("release_ms", 120.0))
        mode = sc.get("mode", "broadband")
        low_band_hz = float(sc.get("low_band_hz", 150.0))
        trigger_bus = self.buses.get(trig_role)
        if trigger_bus is None:
            self.report["diagnostics"].append(f"Sidechain trigger '{trig_role}' not found; skipping")
            return

        trigger_audio = trigger_bus.audio
        if mode == "low_band":
            trigger_audio = split_low_high(trigger_audio, low_band_hz, self.sr)[0]
        env = envelope_follower(np.mean(trigger_audio, axis=0), self.sr, attack_ms, release_ms, normalize=True)
        gain = np.vstack([1.0 - amount * env, 1.0 - amount * env])

        for role in targets:
            bus = self.buses.get(role)
            if bus is None:
                continue
            if mode == "low_band":
                low, high = split_low_high(bus.audio, low_band_hz, self.sr)
                bus.audio = low * gain + high
            else:
                bus.audio *= gain
            bus.stats["sidechained_from"] = trig_role

    def render_mix(self) -> None:
        if not self.buses:
            self.mix = np.zeros((2, 0), dtype=np.float32)
            return
        max_len = max(bus.audio.shape[1] for bus in self.buses.values())
        for bus in self.buses.values():
            if bus.audio.shape[1] < max_len:
                pad = np.zeros((2, max_len - bus.audio.shape[1]), dtype=bus.audio.dtype)
                bus.audio = np.hstack([bus.audio, pad])
        self.mix = np.sum([bus.audio for bus in self.buses.values()], axis=0)

    def _apply_master_tone(self) -> None:
        assert self.mix is not None
        master = self.cfg["master"]
        board: List[Any] = [HighpassFilter(float(master.get("hpf_hz", 30.0)))]
        high_shelf = master.get("high_shelf")
        if high_shelf and abs(float(high_shelf.get("gain_db", 0.0))) > 1e-6:
            board.append(HighShelfFilter(float(high_shelf["freq_hz"]), float(high_shelf["gain_db"])))
        comp = master.get("compressor")
        if comp:
            board.append(
                Compressor(
                    threshold_db=float(comp["threshold_db"]),
                    ratio=float(comp["ratio"]),
                    attack_ms=float(comp["attack_ms"]),
                    release_ms=float(comp["release_ms"]),
                )
            )
        self.mix = run_board(self.mix, board, self.sr, self.chunk)
        if comp:
            makeup_db = float(comp.get("makeup_db", 0.0))
            if abs(makeup_db) > 1e-6:
                self.mix = apply_gain_db(self.mix, makeup_db)

    def _apply_reference_tonal_balance(self) -> None:
        assert self.mix is not None
        if self.reference is None:
            return
        match_cfg = self.cfg["master"].get("reference_match", DEFAULT_REFERENCE_MATCH)
        mode = match_cfg.get("tonal_balance_mode", "off")
        if mode == "off":
            return
        reference_audio = self._get_reference_audio()
        assert reference_audio is not None  # reference is not None, so cache returns audio
        mix_tilt = compute_spectral_tilt(self.mix, self.sr)
        ref_tilt = compute_spectral_tilt(reference_audio, self.sr)
        delta = ref_tilt - mix_tilt
        delta *= float(match_cfg.get("amount", DEFAULT_REFERENCE_MATCH["amount"]))
        max_tilt = float(match_cfg.get("max_tilt_db_per_octave", DEFAULT_REFERENCE_MATCH["max_tilt_db_per_octave"]))
        delta = max(-max_tilt, min(max_tilt, delta))
        self.report["reference_tilt_db_per_octave"] = {
            "mix": mix_tilt,
            "reference": ref_tilt,
            "delta": delta,
            "mode": mode,
        }
        if mode == "apply":
            self.mix = apply_spectral_tilt(self.mix, self.sr, delta)

    def _apply_master_multiband(self) -> None:
        assert self.mix is not None
        master = self.cfg["master"]
        multiband = master.get("multiband", {})
        if not multiband.get("enabled"):
            return
        if not HAVE_SCIPY:
            self.report["diagnostics"].append("Multiband requested but scipy is unavailable; skipping")
            return
        bands = [float(b) for b in multiband["bands"]]
        ratios = [float(r) for r in multiband["ratios"]]
        threshold_db = float(multiband.get("threshold_db", -20.0))
        attack_ms = float(multiband.get("attack_ms", 10.0))
        release_ms = float(multiband.get("release_ms", 120.0))

        left, right = self.mix[0], self.mix[1]
        band_signals: List[np.ndarray] = []
        low_l = lr4_lowpass(left, bands[0], self.sr)
        low_r = lr4_lowpass(right, bands[0], self.sr)
        band_signals.append(np.vstack([low_l, low_r]))
        for i in range(len(bands) - 1):
            lo, hi = bands[i], bands[i + 1]
            mid_l = lr4_lowpass(lr4_highpass(left, lo, self.sr), hi, self.sr)
            mid_r = lr4_lowpass(lr4_highpass(right, lo, self.sr), hi, self.sr)
            band_signals.append(np.vstack([mid_l, mid_r]))
        high_l = lr4_highpass(left, bands[-1], self.sr)
        high_r = lr4_highpass(right, bands[-1], self.sr)
        band_signals.append(np.vstack([high_l, high_r]))

        out = np.zeros_like(self.mix)
        for band, ratio in zip(band_signals, ratios):
            compressor = Compressor(
                threshold_db=threshold_db,
                ratio=ratio,
                attack_ms=attack_ms,
                release_ms=release_ms,
            )
            out += run_board(band, [compressor], self.sr, self.chunk)
        self.mix = out

    def _apply_loudness_and_reference(self) -> None:
        assert self.mix is not None
        master = self.cfg["master"]
        target_lufs = float(master.get("target_lufs", -14.0))
        if HAVE_LOUDNORM and self.reference is not None:
            reference_audio = self._get_reference_audio()
            assert reference_audio is not None
            ref_lufs = pyln.Meter(self.sr).integrated_loudness(reference_audio.T.astype(np.float32))
            self.report["reference_lufs"] = ref_lufs
            target_lufs = ref_lufs

        pre_lufs = float("nan")
        post_lufs = float("nan")
        if HAVE_LOUDNORM:
            meter = pyln.Meter(self.sr)
            pre_lufs = meter.integrated_loudness(self.mix.T.astype(np.float32))
            self.mix = apply_gain_db(self.mix, target_lufs - pre_lufs)
            post_lufs = meter.integrated_loudness(self.mix.T.astype(np.float32))
        else:
            self.report["diagnostics"].append("pyloudnorm unavailable; skipping LUFS normalization")
        self.report["master_lufs"] = {"target": target_lufs, "pre": pre_lufs, "post_prelimit": post_lufs}

    def _apply_final_limiter(self) -> None:
        assert self.mix is not None
        limiter_cfg = {**DEFAULT_LIMITER, **self.cfg["master"].get("limiter", {})}
        ceiling_dbfs = float(limiter_cfg["ceiling_dbfs"])
        release_ms = float(limiter_cfg["release_ms"])
        self.mix = apply_peak_limiter(self.mix, self.sr, ceiling_dbfs, release_ms)
        self.mix = safe_peak_normalize(self.mix, peak_dbfs=ceiling_dbfs)
        self.report["master_limiter"] = {
            "ceiling_dbfs": ceiling_dbfs,
            "release_ms": release_ms,
            "post_peak_dbfs": calc_peak_dbfs(self.mix),
        }
        if HAVE_LOUDNORM:
            meter = pyln.Meter(self.sr)
            self.report["master_lufs"]["post"] = meter.integrated_loudness(self.mix.T.astype(np.float32))

    def master_chain(self) -> None:
        assert self.mix is not None
        self._apply_master_tone()
        self._apply_reference_tonal_balance()
        self._apply_master_multiband()
        self._apply_loudness_and_reference()
        self._apply_final_limiter()

    def print_balance(self) -> None:
        print("\nBus Balance:")
        for role, bus in sorted(self.buses.items()):
            rms = calc_rms_dbfs(bus.audio)
            bar = "#" * max(0, int((rms + 35) * 0.8))
            print(f"  {role:8s}: {bar} {rms:.1f} dBFS ({len(bus.stems)} stems)")

    def save(self, out_path: Path, bitdepth: int) -> None:
        assert self.mix is not None
        subtype = "PCM_24" if bitdepth == 24 else ("PCM_16" if bitdepth == 16 else "FLOAT")
        sf.write(str(out_path), self.mix.T, self.sr, subtype=subtype)
        print(f"Wrote mix: {out_path} | SR={self.sr} BitDepth={bitdepth}")

    def save_report(self, out_path: Optional[Path]) -> Path:
        if out_path is None:
            report_path = Path("analysis_report.txt")
        else:
            report_path = out_path.with_name(out_path.stem + "_report.txt")
        with report_path.open("w", encoding="utf-8") as handle:
            handle.write("Stem Mixer Report\n")
            handle.write("=" * 72 + "\n\n")
            handle.write("Input stems:\n")
            for stem in self.stems:
                handle.write(f"- {stem.path.name} -> role={stem.role}\n")
                handle.write(
                    f"  input_rms={stem.input_stats['input_rms_dbfs']:.2f} dBFS, "
                    f"input_peak={stem.input_stats['input_peak_dbfs']:.2f} dBFS, "
                    f"leading_silence={stem.input_stats['leading_silence_ms']:.1f} ms\n"
                )
                if "post_rms_dbfs" in stem.stats:
                    handle.write(
                        f"  processed_rms={stem.stats['post_rms_dbfs']:.2f} dBFS, "
                        f"processed_peak={stem.stats['post_peak_dbfs']:.2f} dBFS, "
                        f"deesser={stem.stats.get('deesser_mode', 'off')}\n"
                    )
            handle.write("\nRole counts:\n")
            for role, count in sorted(self.report.get("role_counts", {}).items()):
                handle.write(f"- {role}: {count}\n")
            if self.buses:
                handle.write("\nBuses:\n")
                for role, bus in sorted(self.buses.items()):
                    handle.write(
                        f"- {role}: stems={bus.stats.get('stem_count', len(bus.stems))}, "
                        f"rms={bus.stats.get('post_rms_dbfs', float('nan')):.2f} dBFS, "
                        f"peak={bus.stats.get('post_peak_dbfs', float('nan')):.2f} dBFS\n"
                    )
            if "master_lufs" in self.report:
                handle.write("\nMaster loudness:\n")
                master_lufs = self.report["master_lufs"]
                handle.write(
                    f"- target={master_lufs.get('target')}\n"
                    f"- pre={master_lufs.get('pre')}\n"
                    f"- post_prelimit={master_lufs.get('post_prelimit')}\n"
                    f"- post={master_lufs.get('post')}\n"
                )
            if "master_limiter" in self.report:
                limiter = self.report["master_limiter"]
                handle.write("\nLimiter:\n")
                handle.write(
                    f"- ceiling_dbfs={limiter['ceiling_dbfs']}\n"
                    f"- release_ms={limiter['release_ms']}\n"
                    f"- post_peak_dbfs={limiter['post_peak_dbfs']}\n"
                )
            if "reference_tilt_db_per_octave" in self.report:
                tilt = self.report["reference_tilt_db_per_octave"]
                handle.write("\nReference tonal balance:\n")
                handle.write(
                    f"- mode={tilt['mode']}\n"
                    f"- mix_tilt={tilt['mix']}\n"
                    f"- reference_tilt={tilt['reference']}\n"
                    f"- applied_delta={tilt['delta']}\n"
                )
            handle.write("\nDiagnostics:\n")
            for item in self.report.get("diagnostics", []):
                handle.write(f"- {item}\n")
        print(f"Report saved: {report_path}")
        return report_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automatic stem mixer with YAML config")
    parser.add_argument("--in", dest="in_dir", required=True, help="Folder containing stems")
    parser.add_argument("--out", dest="out_path", default=None, help="Output WAV path")
    parser.add_argument("--config", dest="config_path", required=True, help="YAML config path")
    parser.add_argument("--sr", dest="sr", type=int, default=None, help="Override sample rate from config")
    parser.add_argument("--bitdepth", type=int, default=24, choices=[16, 24, 32])
    parser.add_argument("--chunk_samples", type=int, default=262144)
    parser.add_argument("--reference", dest="reference", default=None, help="Optional reference WAV/AIFF/FLAC")
    parser.add_argument("--analyze-only", action="store_true", help="Analyze stems and write report without rendering mix audio")
    return parser.parse_args()


def main() -> None:
    require_runtime_dependencies()
    args = parse_args()
    in_dir = Path(args.in_dir)
    out_path = Path(args.out_path) if args.out_path else None
    cfg_path = Path(args.config_path)
    reference = Path(args.reference) if args.reference else None

    files = sorted([path for ext in ("*.wav", "*.aif", "*.aiff", "*.flac") for path in in_dir.glob(ext)])
    if not files:
        print("No stems found.", file=sys.stderr)
        raise SystemExit(1)
    if out_path is None and not args.analyze_only:
        raise SystemExit("--out is required unless --analyze-only is used")

    cfg = load_config(cfg_path)
    mixer = Mixer(
        files=files,
        cfg=cfg,
        sr_cli=args.sr,
        chunk_samples=args.chunk_samples,
        reference=reference,
        analyze_only=args.analyze_only,
    )

    print(f"Found {len(mixer.stems)} stems:")
    for stem in mixer.stems:
        print(f"  - {stem.path.name} -> {stem.role}")

    if args.analyze_only:
        mixer.save_report(out_path)
        return

    mixer.process_stems()
    mixer.build_buses()
    mixer.apply_sidechain()
    mixer.render_mix()
    mixer.print_balance()
    mixer.master_chain()
    mixer.save(out_path, bitdepth=args.bitdepth)
    mixer.save_report(out_path)


if __name__ == "__main__":
    main()
