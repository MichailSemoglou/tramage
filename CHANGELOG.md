# Changelog

All notable changes to tramage are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project follows [Semantic Versioning 2.0.0](https://semver.org/).

## [0.2.0] - 2026-10-14

First public release.

### Added

- Matrix-valued Damera–Venkata error diffusion (`tramage.dv_filter`):
  a training-free, closed-form filter design in Oklab, and its evaluation
  arms in the kb_baseline experiment (`analyze_dv.py`; the full 124-image,
  six-configuration corpus comparison).
- Input validation with clear `ValueError` messages at the public
  boundaries: `diffuse` and `kb_diffuse` reject malformed images and empty
  or non-finite palettes; `lowfreq_energy_fraction` and
  `spectral_anisotropy` reject out-of-range channel indices.
- `Raises` sections across the public API documenting every deliberate
  error contract.
- `tests/test_quickstart.py`: the documented usage example runs in CI.
- `py.typed` marker for downstream type checking (PEP 561).

### Changed

- The package version now has a single source of truth:
  `pyproject.toml` reads it from `tramage.__version__` via hatchling.
- CI split into `lint`, `test` (Python 3.10–3.12), and `build-check` jobs,
  plus a weekly scheduled run; a tag-triggered PyPI publish workflow.
- The test suite promotes warnings to errors and reports the ten slowest
  tests; branch coverage is configured with a 50% floor.

## [0.1.0] - 2026-07-10

### Added

- Initial release: palette-structure-aware error diffusion in Oklab.
- Global and per-cell shaping matrices from palette covariance
  eigenstructure.
- Vector Floyd–Steinberg diffusion (`tramage.diffusion`) and the
  Kolpatzik–Bouman scalar HVS filter (`tramage.kb_filter`).
- Metric suite: S-CIELAB ΔE, systematic bias, noise, blurred tone error,
  low-frequency energy fraction, spectral anisotropy
  (`tramage.metrics`).
- Palette extraction and eigenspectrum reporting (`tramage.palette`).
- Color-space conversions: sRGB, Oklab, CIELAB, XYZ (`tramage.colour`).
- Swatch loaders: hex lists, GIMP GPL, Adobe ASE (`tramage.swatches`).
- Seeded, crash-safe experiment harnesses under `experiments/`.
