# Tramage

Tramage is palette-structure-aware error diffusion dithering. Give it an image and a palette, and it dithers as ordinary vector Floyd-Steinberg would, except that the error carried between pixels is first reshaped along the palette’s own geometry in Oklab, computed once per palette, so error is passed through where the palette has real extent and damped where it does not. If your palette is a family of related tones rather than a broad spread of color, and dithered output has always looked grainier or noisier than it should, this is built for that case.

![Near-degenerate palette: baseline shows visible noise in the center crop that the shaped variants damp](https://raw.githubusercontent.com/MichailSemoglou/tramage/main/gallery/images/near-degenerate.png)

## What it does not do

Tramage does not invent colors the palette lacks. Given a color the palette cannot represent exactly, both this method and ordinary error diffusion produce the same kind of approximation; what changes is where the resulting, unavoidable noise is allowed to sit. Directed away from directions the palette has no room for, it tends to show up as finer, quieter noise rather than the slow, low-frequency drift that unshaped diffusion leaves behind.

## Installation

```bash
pip install tramage
```

See docs/quickstart.md for a minimal working example, and docs/for-designers.md for a plain-language explanation of why this matters for palette-based work in particular.

## Gallery

![All three variants produce visually identical output on an exactly rank-deficient palette](https://raw.githubusercontent.com/MichailSemoglou/tramage/main/gallery/images/hero-rank-deficient.png)

The image above is a null result, shown honestly: on a palette that is exactly confined to a single plane in Oklab, vector Floyd-Steinberg and shaped diffusion produce identical output. This is the inertness lemma (see docs/mathematical-appendix.md): error orthogonal to a palette’s affine hull can never change a nearest-neighbor decision, so there is nothing for shaping to protect against. See gallery/README.md for the full set of comparisons, including error spectra and the numbers behind the noise reduction shown above.

## How it works

Standard vector error diffusion propagates the full quantization error vector to neighboring pixels. This package instead propagates $e’ = Me$, where $M$ is derived from the palette’s own covariance in Oklab: eigendecompose the covariance, shrink each eigenvalue to a gain of $\lambda / (\lambda + \varepsilon)$, and reassemble. Axes where the palette has real extent keep a gain near 1; near-degenerate axes are damped toward 0. The result is a single global matrix, closed-form and computed once per palette; dithering itself costs one 3×3 matrix multiply per pixel beyond ordinary error diffusion.

A per-cell variant — one matrix per palette color, computed from its nearest neighbors — is also implemented and exported, but is **not part of the recommended method**. On the 124-image corpus run (see `experiments/dithering_quality/`) it holds up on noise reduction but loses on both tone accuracy and the low-frequency worm metric from K=16 onward; natural-image k-means palettes are globally degenerate enough that the local variant's added complexity does not pay off. It remains in the package as the historical record of an explored extension, not as a second contribution.

For reference, the package also ships a matrix-valued Damera–Venkata filter (`dv_filters`, `dv_diffuse`), the exact prior art in this space. On the 124-image corpus it is the strongest unshaped baseline tested, and shaped Floyd–Steinberg matches or beats it at the small-to-moderate palette sizes the method targets (see `experiments/kb_baseline/`).

## Limitations

This is an honest account of where the method is weak, not just where it is strong.

- It offers nothing on palettes with real extent in every direction. For a broad, roughly isotropic palette, the shaping matrix approaches the identity and output is indistinguishable from ordinary vector Floyd-Steinberg.
- On palettes that are close to, but not exactly, degenerate, shaping is a trade-off rather than a free improvement: it reduces noise at some cost to fidelity in directions the palette can barely represent. The epsilon parameter controls this trade, and choosing it well requires some judgment about the palette at hand. On the Kodak subset this trade is free for K ≤ 64 and appears at K ≥ 128, where the palette has enough colors to start representing the formerly-flat direction; on CQ100 no tone cost appears at any K.
- At K = 8 on the Kodak subset, shaping raises Floyd-Steinberg's low-frequency ("worm") error by 52% (median of per-image ratios; pooled means give +41%), while on CQ100 it lowers it by 32%; the full-corpus median is −23%. The sign reversal is the convergence mechanism, not a caveat: shaping pulls low-frequency error toward a common level, so images beginning below that level rise and those beginning above it fall. Per-K detail: `experiments/kb_baseline/`.
- The evaluation spans 124 images across two stock-photo corpora (Kodak, CQ100), but stock photography is still not designer practice. A broader run (real designer palettes; other test sets) is the natural next step; see `experiments/README.md`.
- Nothing here addresses palette choice itself. A palette that is a poor fit for an image will still dither poorly; this method only changes what happens to the error once quantization has already committed to that palette.

## Repository structure

- `src/tramage/`: the package. `colour.py` (sRGB and Oklab conversion), `shaping.py` (matrix construction), `diffusion.py` (serpentine error diffusion), `metrics.py` (fidelity and noise measures), `palette.py` (palette extraction and eigenspectrum reporting).
- `prototypes/`: the original research scripts, kept verbatim as the historical record of the work.
- `experiments/`: the evaluation harness and results — 124 images (Kodak + CQ100): dithering quality, filter baselines (KB/DV), palette geometry, color space.
- `gallery/`: before-and-after comparisons, reproducible using only the installed package (no prototype code).
- `docs/`: a quickstart, a plain-language explanation for designers, and the mathematical appendix.
- `PREREGISTRATION.md`: the analysis plan as written. The numbers it cites (e.g. the 45% worm cost) predate the 124-image re-run and are kept unchanged as the historical record; treat any discrepancy with `experiments/` results as provenance, not error.
- `tests/`: round-trip color conversion, the isotropic limit, the inertness lemma, and energy conservation of the diffusion kernel.

## Citing

See [CITATION.cff](CITATION.cff).

## Licence

MIT. See [LICENSE](LICENSE).
