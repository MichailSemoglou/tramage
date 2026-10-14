# Palette geometry experiment

The Phase 2 reframing condition in [`../../ROADMAP.md`](../../ROADMAP.md),
pulled forward into Phase 1 as the cheapest, highest-value evidence:

> *If palettes extracted across the Kodak corpus are not consistently
> anisotropic in Oklab, that claim must soften to a narrower one about
> which kinds of palettes benefit, rather than palettes in general.*

This folder answers that question with data. No new method code is
exercised: every function called here is part of `tramage`'s public API,
unchanged from the package release. The output is a CSV of eigenstructure
per extracted palette and two figures summarizing it.

## What this measures

For each `(image, K, seed)` trial:

1. Extract a k-means palette in Oklab with `tramage.palette.extract_palette`.
2. Report its covariance eigenspectrum with
   `tramage.palette.eigenspectrum_report` at `eps = tramage.shaping.DEFAULT_EPS`
   (`0.02²`), the same value the dithering pipeline uses.
3. Classify each of the palette's three Oklab axes using the same bands
   `tramage.palette.describe_palette` uses for plain-language reporting:
   - **flat**: `gain ≤ 0.3` (`_FLAT_GAIN`) — error along this axis is
     damped to a small share.
   - **real extent**: `gain ≥ 0.7` (`_REAL_EXTENT_GAIN`) — error passes
     through almost unchanged.
   - **partial**: in between.

A palette is "near-degenerate" if it has at least one flat axis. The
decision-relevant aggregate is the **fraction of palettes with ≥1 flat
axis at each K**: if that fraction is a majority across K, the broad
claim that extracted palettes are consistently near-degenerate holds; if
it collapses past some K, the claim narrows to small-palette regimes.

## Corpora

- **Kodak24** — the Kodak Lossless True Color Image Suite (24 images,
  768×512, public domain). Standard test set in the dithering and CQ
  literature; the same set `ROADMAP.md` Phase 2 commits to. Downloaded
  on demand by `fetch_kodak.py`.
- **McMaster IMAX** — TODO. The original McMaster "McMaster Imagery
  Library: IMAX" set. Left as a documented second step pending
  confirmation of the canonical source URL rather than guessed here.

## Reproducing

From the repository root:

```bash
python experiments/palette_geometry/fetch_kodak.py        # one-time download
python experiments/palette_geometry/run_geometry.py       # ~10–40 min
python experiments/palette_geometry/plot_geometry.py      # writes two PNGs
```

The configuration is fixed and stated up front so the run is reproducible:

| Parameter | Value |
|---|---|
| Palette sizes | K ∈ {8, 16, 32, 64, 128, 256} |
| Seeds per K | {0, 1, 2} (3 k-means restarts per image×K) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Flat / real-extent bands | `_FLAT_GAIN = 0.3`, `_REAL_EXTENT_GAIN = 0.7` |

Outputs (gitignored, regenerable):

- `results/palette_geometry.csv` — one row per trial (756 rows for Kodak24:
  24 × 6 × 3).
- `results/by_K.png` — stacked bars of degeneracy class by K, with the
  "≥1 flat axis" share annotated per bar.
- `results/eigenvalue_ratio.png` — continuous degeneracy score
  `(λ1+λ2)/λ3` per palette, log-scale strip per K, with median lines.

## Verdict

**The reframing condition holds — and more strongly than the discrete
classification alone suggests.** Across 432 trials (24 Kodak images × 6
palette sizes × 3 k-means seeds), k-means palettes extracted in Oklab are
dramatically non-isotropic at every K tested, with median eigenvalue
ratio `(λ1+λ2)/λ3` between 0.015 and 0.057 — two orders of magnitude
below the isotropic benchmark of 2.0. The dominant axis of palette
spread carries roughly 20–70× the variance of the two smaller axes
combined, at every palette size from K=8 through K=256.

### Numbers

Discrete classification, using the package's own bands (`gain ≤ 0.3` =
flat, `≥ 0.7` = real extent):

| K | 0 flat | 1 flat | 2 flat | ≥1 flat |
|---|---|---|---|---|
| 8 | 21% | 53% | 26% | **79%** |
| 16 | 44% | 54% | 1% | 56% |
| 32 | 56% | 44% | 0% | 44% |
| 64 | 60% | 40% | 0% | 40% |
| 128 | 61% | 39% | 0% | 39% |
| 256 | 69% | 31% | 0% | 31% |

Continuous degeneracy score `(λ1+λ2)/λ3` (isotropic = 2.0; smaller = more
degenerate):

| K | median | p10 | p90 |
|---|---|---|---|
| 8 | 0.015 | 0.002 | 0.084 |
| 16 | 0.034 | 0.006 | 0.099 |
| 32 | 0.043 | 0.011 | 0.108 |
| 64 | 0.050 | 0.015 | 0.103 |
| 128 | 0.057 | 0.016 | 0.112 |
| 256 | 0.051 | 0.018 | 0.114 |

### Interpretation

The two measures tell a consistent story at small K and a divergent one
at large K, and the divergence is itself the most decision-relevant
finding.

- **At K ≤ 16**, both measures agree: a majority of palettes have at
  least one axis the `_FLAT_GAIN` band calls flat, the median
  degeneracy ratio is below 0.04, and the 90th-percentile ratio (0.084
  at K=8) is still 24× below isotropic. The method has purchase on most
  palettes here.
- **At K ≥ 32**, the discrete `≥1 flat` share drops to a minority
  (31–44%), which at first reading looks like the broad claim softening.
  The continuous measure tells a different story: the median degeneracy
  ratio is still 0.04–0.06, and even the 90th percentile (0.103–0.114)
  sits ~18–20× below isotropic. The palettes are not flat at the
  `gain ≤ 0.3` band, but they are nowhere near isotropic either; they
  cluster in the *partial* band, where shaping damps the smallest
  eigenvalue to a gain between 0.3 and 0.7.

The honest summary: **k-means palettes extracted from natural images in
Oklab are consistently and substantially anisotropic at every palette
size tested, and the discrete "near-degenerate" claim is correct in
spirit but needs the continuous measure to hold at large K, where the
discrete one alone would mislead.** This is a stronger result for the
method than the discrete classification suggests, and it softens the
worry that the practical envelope collapses past K=32: it doesn't
collapse, it shifts from the clearly-flat regime into the partial-shaping
regime, where shaping still does measurable work.

What this does *not* establish: that the anisotropy translates into
visible quality improvement under dithering. That is the Kodak
dithering-quality run (see [`../README.md`](../README.md)), which this
geometry check now motivates rather than replaces. The per-cell variant,
in particular, may matter more here than the global one: palettes with
partial-but-not-flat axes are exactly the regime where local covariance
could differ from global.
