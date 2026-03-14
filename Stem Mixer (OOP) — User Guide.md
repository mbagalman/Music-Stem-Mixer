# Stem Mixer Pro (OOP) — User Guide

> A beginner-friendly, step-by-step guide for mixing and mastering your stems with free, open-source tools.

---

## What this tool does

You give the script a **folder of stems** (separate audio files like vocals, drums, bass…). It will:

1. **Load and clean** each stem (high-pass filter, optional de-essing, EQ peaks, compression).
2. **Set stereo width** per stem (narrow bass/vocals, wider guitars/keys).
3. **(Optional) Sidechain duck** (e.g., kick/drums make bass/keys dip slightly).
4. **Sum to a mix** and run a **master bus** (HPF, gentle EQ/comp, optional multiband, limiter).
5. **Target a loudness** (LUFS) or match a **reference track** (optional).
6. Write the **final WAV** and a **mix report** with useful stats.

It’s designed to be:

* **Simple for beginners** (sensible defaults).
* **Safe for your computer** (chunked/streamed processing; won’t hog RAM).
* **Customizable** via a human-readable **YAML config** file.

---

## Download the files

* Canonical script:
  **[stem\_mixer\_oop.py](./stem_mixer_oop.py)**
* Baseline configuration:
  **[config.yaml](./config.yaml)**
* Optional vocal de-esser preset:
  **[config\_with\_deesser.yaml](./config_with_deesser.yaml)**

> This directory now uses a single canonical script. Start with `config.yaml`, then switch to the de-esser preset if you want external vocal de-essing.

---

## System requirements

* **Python 3.9+** (3.10/3.11 recommended)
* Works on **Windows**, **macOS**, **Linux**

### Audio formats supported

Input stems: `.wav`, `.aif`, `.aiff`, `.flac`
(If you only have MP3s, convert them to WAV first for best quality.)

---

## Install the dependencies (one-time)

Open a terminal (PowerShell on Windows) and run:

```bash
# 1) Create and activate a virtual environment (recommended)
python -m venv venv
# macOS/Linux:
source venv/bin/activate
# Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 2) Install packages
pip install -r requirements.txt
```

> Don’t worry—if some optional packages are missing (e.g., `pyloudnorm` or `scipy`), the script still runs; it just disables the related features and tells you why.

---

## Prepare your stems

1. Put all audio files for **one song** into a folder, e.g. `./stems`.
2. **Name stems clearly** so roles can be auto-detected:

   * `vocals_lead.wav`, `vox_bv.wav` → **vocals**
   * `drums.wav`, `kick.wav`, `kit.aif` → **drums**
   * `bass.flac` → **bass**
   * `gtr_left.wav`, `guitar_1.wav` → **guitar**
   * `keys.wav`, `piano.aif`, `synth_pad.flac` → **keys**
3. If you’re unsure, the “default” chain will be used.

> Tip: All stems should start at the same time (bar 1) so they line up correctly.

---

## Quick start

Put `stem_mixer_oop.py` and `config.yaml` alongside your `stems` folder, then run:

```bash
# macOS/Linux
python stem_mixer_oop.py --in ./stems --out ./mix.wav --config ./config.yaml

# Windows (PowerShell)
python .\stem_mixer_oop.py --in .\stems --out .\mix.wav --config .\config.yaml
```

You’ll see logs as it processes, plus a **mix report** next to your output:

```
mix.wav
mix_report.txt
```

---

## Command-line options

```bash
python stem_mixer_oop.py \
  --in ./stems \                # folder containing your stems
  --out ./mix.wav \             # output file path (WAV)
  --config ./config.yaml \      # YAML config file
  --sr 48000 \                  # (optional) override sample rate
  --chunk_samples 262144 \      # (optional) processing block size
  --bitdepth 24 \               # 16 | 24 | 32 (float)
  --reference ./ref.wav \       # (optional) match reference loudness
  --analyze-only                # (optional) preflight report, no WAV render
```

**When to tweak options**

* `--sr`: If your stems are mismatched, set a target (44100 or 48000 are common).
* `--chunk_samples`: Lower this if you run out of memory; raise it for speed.
* `--bitdepth`: Use 24 for high-quality; 16 if you specifically need CD standard.
* `--reference`: Matches loudness and can optionally report/apply spectral tilt guidance.
* `--analyze-only`: Loads stems, checks alignment/hotness/clipping/role counts, and writes the report without rendering audio.

---

## Understanding `config.yaml`

This file is your “mix recipe.” You can set per-stem behavior, role buses, sidechain, alignment, and master settings.

### 1) Stem role detection

```yaml
stem_detection:
  vocals: [vox, vocal, lead, singer]
  drums:  [drum, drums, kit, percussion]
  bass:   [bass, sub]
  guitar: [gtr, guitar]
  keys:   [keys, piano, synth, pad, rhodes, organ]
```

> If a filename contains any of these keywords, it’s assigned that role. Otherwise: `default`.

### 2) Per-stem chains

Each role inherits a chain like:

```yaml
chains:
  vocals:
    hpf_hz: 80.0                # high-pass filter cutoff (rumble/tone cleanup)
    stereo_width: 0.8           # 1.0 keeps original; <1 narrows; >1 widens
    deesser:
      mode: builtin             # off | builtin | external
      frequency_hz: 6200.0
      threshold_db: -28.0
      ratio: 4.0
      amount: 0.6
    eq_peaks:
      - { freq: 300.0, q: 1.0, gain_db: -2.0 }   # notch mud
      - { freq: 3500.0, q: 1.2, gain_db: -1.5 }  # tame harshness
    compressor:
      threshold_db: -18.0
      ratio: 2.5
      attack_ms: 10.0
      release_ms: 80.0
      makeup_db: 0.0
    parallel_comp_blend: 0.0    # 0..1 (NY comp blend; drums often ~0.3)
```

**Guidelines**

* **Vocals/bass**: keep **narrow** (`stereo_width` ≤ 1.0) for center focus.
* **Guitars/keys/drums**: **wider** (1.0–1.3) adds space.
* Use **EQ peaks** sparingly (±1–2 dB) unless fixing a clear problem.
* **Parallel comp**: great on **drums** for punch (`0.2–0.4`).

### 3) Role buses

The mixer now sums stems by role first, then balances the **bus** instead of normalizing each file independently.

```yaml
buses:
  vocals:
    target_rms_dbfs: -18.5
    gain_db: 0.0
    pan: 0.0
    stereo_width: 0.95

  bass:
    target_rms_dbfs: -19.5
    pan: 0.0
    stereo_width: 0.8
    mono_below_hz: 160.0
```

This prevents stacked guitars, harmonies, or percussion layers from getting louder just because there are more files.

### 4) Sidechain ducking (optional)

```yaml
sidechain:
  enabled: false
  trigger: drums        # envelope source, e.g., "drums"
  targets: [bass, keys] # buses to duck
  amount: 0.3           # 0..1 (higher = more duck)
  attack_ms: 5.0
  release_ms: 120.0
  mode: low_band        # broadband | low_band
  low_band_hz: 150.0
```

**Why use it?** Helps the kick and snare cut through by gently ducking bass/keys.
**EDM pop tip:** `amount: 0.4–0.6`, `attack_ms: 1–5`, `release_ms: 100–200`.

### 5) Master chain

```yaml
master:
  hpf_hz: 30.0
  high_shelf: { freq_hz: 10000.0, gain_db: 1.0 }  # gentle air; set 0 to disable
  compressor: { threshold_db: -14.0, ratio: 1.4, attack_ms: 15.0, release_ms: 150.0, makeup_db: 0.0 }
  limiter:
    ceiling_dbfs: -1.0
    release_ms: 120.0
  target_lufs: -14.0                               # streaming-friendly loudness
  reference_match:
    tonal_balance_mode: report                     # off | report | apply
    amount: 0.5
    max_tilt_db_per_octave: 3.0
  multiband:
    enabled: false
    bands: [100, 500, 2000, 8000]                  # crossovers (Hz)
    ratios: [1.5, 1.8, 2.0, 1.5, 1.3]              # N+1 ratios per band
    threshold_db: -20.0
    attack_ms: 10.0
    release_ms: 120.0
```

* **High-shelf** adds a touch of “air.” Set `gain_db: 0` to disable.
* **Compressor** is gentle “glue.” Avoid heavy ratios on the master.
* **Multiband** (needs SciPy) splits the spectrum into bands using **Linkwitz–Riley** crossovers for flatter summing—use sparingly.

---

## Example workflows (“recipes”)

### Pop / Singer-Songwriter (clear vocal, natural dynamics)

* Vocals: keep the bus near `target_rms_dbfs: -18.5`, builtin or external de-esser on, small EQ cuts at 300–400 Hz.
* Drums: `parallel_comp_blend: 0.25`, `stereo_width: 1.0`.
* Bass: `stereo_width: 0.5`, HPF 30 Hz.
* Master: `target_lufs: -14`, `high_shelf: +0.5 to +1.0 dB`.

### Indie Rock (wide guitars, punchy drums)

* Guitars: `stereo_width: 1.2–1.3`, slight presence boost around 3–4 kHz.
* Drums: `parallel_comp_blend: 0.3`.
* Sidechain: optional, low `amount: 0.2` on bass.
* Master: `target_lufs: -12 to -14`, gentle comp (ratio 1.3–1.5).

### EDM / Pop-EDM (pump and clarity)

* Sidechain: `enabled: true`, `trigger: drums`, `targets: [bass, keys]`, `amount: 0.4–0.6`, `attack_ms: 2–5`.
* Drums: `parallel_comp_blend: 0.35`.
* Master: consider multiband enabled, `ratios` slightly higher in low-mid.

---

## Output files you’ll get

* **`mix.wav`**: Your final stereo mix (bit depth you chose).
* **`mix_report.txt`**: What was processed, per-stem levels/peaks, master settings, LUFS info.

### Reading the report

* **Per-stem RMS**: sanity-check balance (e.g., vocals around -20 dBFS, drums -18 dBFS).
* **Post-LUFS**: confirms loudness target (or reference match).
* **Ceiling**: final peak normalization after limiting.

---

## Performance tips

* If you see slowdowns or RAM spikes, lower `--chunk_samples`:

  * Start with `262144` (default). Try `131072` or `65536` if needed.
* Multiband uses zero-phase filters (`sosfiltfilt`), which need full buffers—disable it for very long tracks if performance matters.

---

## Troubleshooting

**“No stems found.”**
Check your path and extensions: only `.wav`, `.aif`, `.aiff`, `.flac` are scanned.

**“pyloudnorm not installed; skipping LUFS normalization.”**
Install it: `pip install pyloudnorm` (or just ignore—mix will still render).

**“Multiband requested but SciPy not installed; skipping.”**
Install it: `pip install scipy`, or set `multiband.enabled: false`.

**Output sounds distorted / too loud**

* Lower the relevant bus `target_rms_dbfs` (e.g., vocals from -18.5 → -20).
* Reduce master comp ratio or raise threshold.
* Ensure `master.limiter.ceiling_dbfs` is -1.0 or lower.

**Vocals too wide / unfocused**
Set `stereo_width: 0.7–0.9` for vocals; keep low-end instruments narrow.

**Kick/bass conflict**
Enable sidechain with `trigger: drums`, `targets: [bass]`, `amount: 0.3–0.5`.

---

## Frequently asked questions

**Do I need to edit the Python code to use this?**
No—tune everything in `config.yaml`.

**Can I run multiple songs with different settings in one session?**
Yes. Use different config files (e.g., `config_pop.yaml`, `config_edm.yaml`) and run the script per song/folder.

**Will it change my files in place?**
No, it only reads your stems and writes new outputs.

**What loudness should I target?**
For general streaming, `-14 LUFS` is a safe, musical default. For louder genres, you can push to `-12`, but beware of pumping/harshness.

---

## Good practices & mixing tips

* **Gain staging first**: keep bus targets conservative (-18 to -21 dBFS) and let the master do the last dB or two.
* **Small EQ moves**: ±1–2 dB goes a long way; cut before you boost.
* **Reference tracks**: Use `--reference` to match loudness and A/B your tone/space, not just volume.
* **Mono check**: Narrow key elements to keep mono compatibility solid (vocals, bass, kick/snare).

---

## Example sessions

### Minimal run (defaults)

```bash
python stem_mixer_oop.py --in ./stems --out ./mix.wav --config ./config.yaml
```

### Match a reference track

```bash
python stem_mixer_oop.py --in ./stems --out ./mix.wav --config ./config.yaml \
  --reference ./favorite_mix.wav
```

### Low-memory laptop mode

```bash
python stem_mixer_oop.py --in ./stems --out ./mix.wav --config ./config.yaml \
  --chunk_samples 65536
```

---

## What if I want to go deeper?

* Read the canonical script directly:
  **[stem\_mixer\_oop.py](./stem_mixer_oop.py)**
* Clone the YAML for different genres/projects.
* Add more roles to `stem_detection` (e.g., `strings`, `brass`) and corresponding chains.

---

## Recap

1. Install dependencies.
2. Put stems in a folder, name them clearly.
3. Tune `config.yaml` (or use the defaults).
4. Run the script with `--in`, `--out`, and `--config`.
5. Listen to `mix.wav`, read `mix_report.txt`, and iterate.

You’re set! If you want, share your `config.yaml` and I’ll suggest tweaks for your specific style or a particular song.
