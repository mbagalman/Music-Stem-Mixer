# Stem Mixer Improvement Roadmap and Ticket Pack

## Summary

This roadmap prioritizes **mix quality and correctness first**, with a **built-in de-esser fallback** so the tool improves Suno-style stems even without an external plugin. Tickets are ordered from **high value / low effort** toward **lower value / higher effort**.

## Public Interface Changes

- `master.limiter` now owns final limiter settings with `ceiling_dbfs` and `release_ms`.
- `master.limiter_ceil_dbfs` is still accepted as a deprecated compatibility alias.
- `chains.<role>.deesser` supports `mode: off|builtin|external`, plus builtin settings like `frequency_hz`, `threshold_db`, `ratio`, and `amount`.
- `buses` owns role-level balance and staging; legacy `chains.<role>.target_rms_dbfs` is accepted as a deprecated compatibility alias.
- `--analyze-only` writes diagnostics/reporting without rendering mix audio.

## Ordered Ticket Pack

| Ticket | Value | Effort | Status | Summary |
| --- | --- | --- | --- | --- |
| MIX-01 | High | Low | Implemented | Added strict validation, required config guards, and deprecation warnings for legacy limiter/de-esser/bus balance keys. |
| MIX-02 | High | Medium | Implemented | Reordered mastering to tone/compression -> tonal match -> multiband -> loudness -> final limiter -> safety normalization, with a real enforced ceiling. |
| MIX-03 | High | Medium | Implemented | Added native builtin de-esser mode plus optional external plugin mode with fallback diagnostics. |
| MIX-04 | High | Medium | Implemented | Added automated tests for config migration/validation, limiter ceiling behavior, de-essing, bus normalization, and a synthetic integration smoke test. |
| MIX-05 | High | High | Implemented | Introduced role buses with role-level normalization/staging so stacked stems do not blindly get louder. |
| MIX-06 | High | High | Implemented | Added mono-below-crossover bus control and low-band sidechain mode for kick/bass masking control. |
| MIX-07 | Medium | Low | Implemented | Added diagnostics for clipping, silence, hot/quiet stems, role-count stacking, alignment spread, plugin fallback, and analyze-only reporting. |
| MIX-08 | Medium | Medium | Implemented | Added bus-level pan and stereo width controls. |
| MIX-09 | Medium | Medium | Implemented | Added opt-in leading-silence alignment with diagnostics for timing spread. |
| MIX-10 | Medium | High | Implemented | Added reference spectral-tilt report/apply support alongside LUFS matching. |
| MIX-11 | Low | High | Partially implemented | Reduced dependency on full-file `librosa` loads by using `soundfile` + resampling helpers, but full block-streaming for the whole render pipeline is still future work. |

## Test Plan

- Validate configs that should fail: invalid sidechain amount, empty multiband band list, unsorted crossovers, missing `default` chain, deprecated keys.
- Verify mastering invariants: ceiling is respected, limiter runs after loudness gain, and loudness stays within target tolerance when not heavily ceiling-limited.
- Verify bus invariants: one guitar stem vs two guitar stems should not cause a blind doubling of role level after the bus refactor.
- Verify de-esser behavior: builtin mode works with no plugin installed; external mode falls back cleanly.
- Run a manual end-to-end smoke test with local Suno stems after major changes and compare the generated report for LUFS, peaks, alignment, and balance changes.

## Assumptions

- Primary audience is a solo user mixing exported Suno stems into a single stereo WAV.
- Backward compatibility is preserved where practical, with immediate warnings for deprecated config paths.
- Large local WAV stems remain manual test assets and are not required for automated tests.
- Full low-memory block streaming is the one roadmap item still intentionally incomplete after this pass.
