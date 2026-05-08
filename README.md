# Music Stem Mixer

Automatic stem mixing and lightweight mastering for exported song stems, especially Suno-style multitracks.

The project takes a folder of stems and produces a single stereo WAV with:

- role detection from filenames
- per-stem cleanup with HPF, EQ, compression, and de-essing
- role-bus balancing so stacked stems do not get louder just because there are more files
- optional alignment, sidechain ducking, low-end mono control, and stereo placement
- master bus processing with loudness targeting, final limiting, and optional reference-track guidance
- diagnostics and `--analyze-only` reporting

## Quick Start

```bash
python -m venv venv

# macOS/Linux:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

pip install -r requirements.txt

python stem_mixer_oop.py \
  --in ./stems \
  --out ./mix.wav \
  --config ./config.yaml
```

To generate a report without rendering a mix:

```bash
python stem_mixer_oop.py \
  --in ./stems \
  --config ./config.yaml \
  --analyze-only
```

The report is written next to `--out` as `<stem>_report.txt` (e.g. `mix_report.txt`). When `--analyze-only` is used without `--out`, it writes `analysis_report.txt` in the current directory.

## Repository Layout

- [`stem_mixer_oop.py`](./stem_mixer_oop.py): main CLI and processing pipeline
- [`config.yaml`](./config.yaml): default preset
- [`config_with_deesser.yaml`](./config_with_deesser.yaml): external de-esser preferred
- [`config_cleaner_headroom.yaml`](./config_cleaner_headroom.yaml): safer, lower-loudness preset
- [`config_debug_no_deesser.yaml`](./config_debug_no_deesser.yaml): debugging preset with de-essing off
- [`tests/test_stem_mixer_oop.py`](./tests/test_stem_mixer_oop.py): automated tests
- [`Stem Mixer (OOP) — User Guide.md`](./Stem%20Mixer%20%28OOP%29%20%E2%80%94%20User%20Guide.md): fuller usage guide
- [`docs/mixer-improvement-roadmap.md`](./docs/mixer-improvement-roadmap.md): implemented roadmap and follow-up notes

## Main Features

- Bus-based normalization and gain staging
- Built-in or external vocal de-essing
- Optional reference loudness and spectral-tilt guidance
- Low-band sidechaining for kick/bass masking control
- Opt-in leading-silence alignment
- Final limiter with an explicit ceiling

## Tests

Run the automated suite with:

```bash
python -m unittest discover -s tests -v
```

## Notes

- The local `stems/` WAV files are intended for manual testing and are ignored by Git.
- Some processing paths use optional packages such as `pyloudnorm`, `scipy`, and `numba`. The script degrades gracefully when those are unavailable.

## License

MIT. See [`LICENSE`](./LICENSE).
