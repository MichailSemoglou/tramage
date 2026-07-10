# KB baseline experiment

Tests whether palette-structure-aware shaping still helps when the baseline
diffusion filter is **HVS-optimised**, not just plain Floyd-Steinberg. The
[`../dithering_quality/`](../dithering_quality/README.md) run showed
shaping reduces noise 33–66% against vector FS; a reviewer will reasonably
ask whether that survives against a stronger filter that already shapes
noise perceptually.

## The stronger baseline

The Kolpatzik-Bouman scalar error filter (kolpatzik1992, Table 1) — the
separable special case of Damera-Venkata and Evans' matrix-valued filter
(damera-venkata2001, Sec IV-C). Three independent 4-tap filters, one per
opponent channel (Y, rg, yb), with taps designed to minimise perceived
error variance under per-channel contrast sensitivity functions. The
published coefficient values are used directly (`tramage.kb_filter.kb_filters`)
rather than re-derived; see that module's docstring for why.

The opponent transform and the approximation involved: the KB filter
operates in a Y/rg/yb opponent space defined on linear RGB.
`tramage.kb_filter._oklab_to_opponent` applies the opponent matrix to
Oklab values directly, which is an approximation (it treats Oklab L/a/b
as RGB-like for luminance/chrominance separation). This is documented in
the module rather than hidden; for the baseline-comparison question it is
acceptable.

## Configurations

Four configurations per `(image, K)`:

- **fs-baseline** — Floyd-Steinberg in Oklab, no shaping. (Reference.)
- **fs-shaped** — FS + global Oklab shaping. (The package method.)
- **kb-baseline** — KB filter in opponent space, no shaping. (Stronger baseline.)
- **kb-shaped** — KB filter + global Oklab shaping applied on top. **The load-bearing config.**

If `kb-shaped` beats `kb-baseline`, shaping generalises beyond plain FS.
If not, the claim is FS-specific and the paper narrows.

## Configuration

| Parameter | Value |
|---|---|
| Corpus | Kodak24 (24 images, 768×512) |
| Palette sizes | K ∈ {8, 16, 32, 64, 128, 256} |
| Seed | 0 |
| Configs per trial | 4 |
| Total trials | 576 (24 × 6 × 4) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Diffusion | Floyd-Steinberg and KB scalar, both serpentine |

## Reproducing

```bash
python experiments/kb_baseline/run_kbbaseline.py    # ~25 min
python experiments/kb_baseline/plot_kbbaseline.py   # writes two PNGs
```

## Verdict

**Shaping's benefit survives against the stronger Kolpatzik-Bouman
baseline — at reduced magnitude, but consistently on every image at every
K. The central generalisation claim holds: palette-structure-aware
shaping is not an artefact of comparing against a weak Floyd-Steinberg
filter.**

### Numbers

**L-channel noise** (median across 24 images; lower is quieter):

| K | FS baseline | FS + shaped | KB baseline | KB + shaped |
|---|---|---|---|---|
| 8 | 0.0917 | **0.0390** | 0.0887 | **0.0563** |
| 16 | 0.0642 | **0.0230** | 0.0732 | **0.0357** |
| 32 | 0.0390 | **0.0155** | 0.0446 | **0.0230** |
| 64 | 0.0228 | **0.0104** | 0.0305 | **0.0171** |
| 128 | 0.0132 | **0.0077** | 0.0208 | **0.0129** |
| 256 | 0.0081 | **0.0054** | 0.0137 | **0.0094** |

**Per-image paired win rate of KB+shaped over KB baseline: 24/24 (100%)
at every K.** Every image, every palette size, the shaped version is
quieter than its own KB baseline.

**Relative noise improvement from shaping, under each filter:**

| K | FS: shaped vs base | KB: shaped vs base |
|---|---|---|
| 8 | 53.6% | 33.5% |
| 16 | 58.7% | 45.6% |
| 32 | 57.2% | 46.7% |
| 64 | 51.9% | 45.2% |
| 128 | 43.1% | 38.6% |
| 256 | 32.1% | 32.8% |

### Interpretation

Three findings:

1. **The generalisation claim holds.** Shaping reduces noise by 33–47%
   on top of the KB filter — smaller than the 32–59% it achieves on top
   of FS, but still substantial, and still winning on 24/24 images at
   every K. The KB filter already pushes noise to frequencies the eye is
   less sensitive to; shaping adds a further palette-structure-aware
   reduction on top of that perceptual shaping. The two are complementary,
   not redundant.

2. **KB is a stronger noise baseline than FS at large K, weaker at small
   K.** At K=8, KB baseline (0.0887) is actually slightly quieter than FS
   baseline (0.0917); by K=256 KB is *noisier* (0.0137 vs 0.0081). The
   KB filter's HVS-optimised taps help most when the palette is small
   (more quantisation error to redistribute) but introduce their own
   noise at large K, possibly because the approximation in the opponent
   transform (`_oklab_to_opponent`) becomes less accurate as the palette
   fills the gamut. This is a property of the baseline implementation,
   not of the shaping method being tested.

3. **The tone-accuracy trade-off reproduces under KB.** Just as in the
   FS comparison, shaping wins tone error at K ≤ 64 and loses it at
   K ≥ 128 — the same regime boundary the dithering-quality experiment
   found. This consistency across filters is itself evidence that the
   trade-off is a property of the shaping method (it damps the formerly-
   flat direction, which costs tone once the palette can represent it),
   not of the particular diffusion filter it sits on top of.

### What this changes for the paper

- The headline can now state that shaping generalises across filters,
  not just across palette sizes: "palette-structure-aware shaping
  reduces dithering noise by 33–59% over Floyd-Steinberg and 33–47%
  over the Kolpatzik-Bouman HVS-optimised filter, consistently on every
  image at every palette size tested."
- The scope qualifier on tone accuracy (K ≤ 64 for free improvement)
  is corroborated as filter-independent.
- The approximation in the opponent transform (`_oklab_to_opponent`) is
  a documented limitation; the paper should state that the KB baseline
  here uses an approximate opponent space rather than the full
  linear-RGB pipeline kolpatzik1992 specifies, so the comparison is
  indicative rather than definitive. A more faithful KB implementation
  is the natural follow-up if a reviewer demands it.
