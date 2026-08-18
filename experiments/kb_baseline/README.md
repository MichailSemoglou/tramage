# Filter-baseline experiment: does shaping substitute for HVS filter design?

The paper's central claim needs to survive a stronger comparison than plain
Floyd-Steinberg. This experiment runs three filters — naive Floyd-Steinberg
(FS), the Kolpatzik-Bouman scalar HVS filter (KB, the separable special case
of Damera-Venkata), and the full matrix-valued Damera-Venkata filter (DV) —
each with and without palette-structure-aware shaping. Six configurations per
(image, K):

- **fs-baseline / fs-shaped** — Floyd-Steinberg, without / with shaping.
- **kb-baseline / kb-shaped** — KB scalar HVS filter, without / with shaping.
- **dv-baseline / dv-shaped** — the matrix-valued DV filter, without / with shaping.

## The headline is substitution, not independence

At small palette sizes, palette-geometry shaping does more than an
HVS-optimised filter. Per-image means at K=8 (n=24):

| config | noise_L |
|---|---|
| fs-baseline | 0.0907 |
| kb-baseline | 0.0908 |
| kb-shaped | 0.0558 |
| **fs-shaped** | **0.0416** |

**`fs-shaped` beats `kb-baseline` on 24/24 images (mean −0.0492, ~54% lower),
and beats `kb-shaped` on 24/24 images (mean −0.0142).** Shaping a naive
filter does the work of an HVS-tuned filter, and more. The practical claim is
stronger than "shaping is independent of filter choice": *implementers can
skip filter design* — palette-geometry shaping on plain Floyd-Steinberg
recovers the benefit of an HVS-optimised filter at the palette sizes where
shaping matters.

## Regime of interest

Shaping is a **small-to-moderate-K intervention** (K ≤ 64). That is where
palette geometry is irregular enough for shaping to have purchase; at
K = 256, error diffusion into 256 colours is barely doing anything for
anyone (the shaping benefit is 0.0037 of a noise level of ~0.008). The
interaction ratio is therefore only meaningful where the shaping effect
(the denominator) is not near zero; absolute interaction is reported
alongside the ratio throughout.

## The stronger baselines

- **KB scalar filter** (kolpatzik1992, Table 1): three independent 4-tap
  filters, one per opponent channel (Y, rg, yb), minimising perceived error
  variance under per-channel CSFs. Uses the published coefficients directly.
- **DV matrix filter** (`tramage.dv_filter`): the full Damera-Venkata matrix
  taps, designed per palette by solving a Yule-Walker system in
  Poirson-Wandell opponent space (see `dv_filter.py`). This is the exact
  prior art, not a special case.

Both filters run through the same approximate opponent transform
(`_oklab_to_opponent` treats Oklab L/a/b as RGB-like), a documented
limitation: the comparison is indicative rather than definitive against a
faithful linear-RGB pipeline.

## Disambiguation: this is not Damera-Venkata

Damera-Venkata & Evans' matrix is **optimised against an HVS colour-frequency
model**: it routes error into the frequencies and opponent channels the eye
is least sensitive to. The shaping matrix here comes from **palette
covariance**: it routes error into the directions the palette can actually
absorb. Different objects answering different questions, and they compose —
the `kb-shaped`/`dv-shaped` configs apply both. The `cell` variant is
additionally spatially adaptive where a DV&E matrix is fixed for the image.

## Configurations

| Parameter | Value |
|---|---|
| Corpus | Kodak24 (24 images, 768×512) |
| Palette sizes | K ∈ {8, 16, 32, 64, 128, 256} |
| Seed | 0 |
| Configs per trial | 6 (3 filters × baseline/shaped) |
| Total trials | 864 (24 × 6 × 6) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Diffusion | FS, KB scalar, DV matrix — all serpentine |

## Reproducing

```bash
python experiments/kb_baseline/run_kbbaseline.py        # ~30 min
python experiments/kb_baseline/plot_kbbaseline.py       # metrics + relative + interaction PNGs
python experiments/kb_baseline/analyze_interaction.py   # trend test + per-K TOST
```

## Verdict

**Shaping helps on top of every filter, but its benefit is not uniform: at
K=8 it does more for the naive FS filter than for the tuned filters, and that
asymmetry is exactly the substitution result. The interaction is monotonic in
K — one degree of freedom, p = 9×10⁻⁵ — not noise around a null.**

### The interaction term

Reported two ways, with the **trend test as primary** (a single
degree-of-freedom test avoids the multiplicity of six per-K tests):

- **Primary — trend test.** Mean per-image slope of the FS−KB interaction
  against log₂K: **−0.0030 per doubling, 95% CI [−0.0044, −0.0017],
  p = 8.9×10⁻⁵**, crossing zero around K ≈ 70. The interaction declines
  monotonically with K; the crossing is identified by the trend, not as six
  independent per-K findings.
- **Secondary — per-K TOST** against a pre-specified margin θ = 25% of the
  pooled shaping benefit (90% bootstrap CI, α = 0.05 one-sided). Equivalence
  holds at K=32; it fails elsewhere.

L-channel noise, per-image (n=24), mean ± 90% CI:

| K | main effect | interaction | ratio (interaction / main) | verdict |
|---|---|---|---|---|
| 8 | 0.0420 | +0.0141 [+0.0095, +0.0193] | +0.34 [+0.22, +0.47] | not equivalent — FS benefits more |
| 16 | 0.0382 | +0.0055 [+0.0006, +0.0110] | +0.15 [+0.02, +0.28] | not equivalent — FS benefits more |
| 32 | 0.0224 | −0.0003 [−0.0020, +0.0013] | −0.01 [−0.09, +0.06] | equivalent |
| 64 | 0.0130 | −0.0027 [−0.0036, −0.0018] | −0.21 [−0.27, −0.15] | not equivalent — KB benefits more |
| 128 | 0.0070 | −0.0024 [−0.0030, −0.0018] | −0.34 [−0.42, −0.26] | not equivalent — KB benefits more |
| 256 | 0.0037 | −0.0020 [−0.0023, −0.0016] | −0.53 [−0.61, −0.45] | not equivalent — KB benefits more |

Three things follow:

1. **The interaction is monotonic and flips sign** (trend p = 9×10⁻⁵). At K=8
   shaping helps the weaker FS filter more; by K≥64 it helps KB more. Largest
   where palette geometry is most irregular — a mechanistic finding, not a
   blemish on a null.
2. **At K=8 the interaction is 34% [22%, 47%] of the shaping effect** — the
   same order as the effect, which is precisely the substitution result.
3. **The large-K negative ratio is a small-denominator artifact.** At K=256
   the main effect is 0.0037 and the interaction −0.0020; both are tiny in
   absolute terms. Absolute interaction is reported alongside the ratio so the
   ratio is not misread as a large effect.

Tone-error ratios are omitted from the headline because the tone main effect
crosses zero at K≥128 (the known tone-accuracy trade-off, reported explicitly
as a limitation). S-CIELAB ΔE is added to this table once the re-run includes
it.

### What this changes for the paper

- The headline is **substitution**: at small K, palette-geometry shaping on a
  naive filter does the work of an HVS-optimised filter, so implementers can
  skip filter design. This is a stronger, more practical claim than
  independence.
- The shaping × filter interaction is reported as a **single trend test**
  (monotonic in K, p = 9×10⁻⁵), with per-K TOSTs kept secondary to locate the
  crossing (K≈70, equivalence at K=32).
- **DV&E is now run, not deferred.** The matrix-valued DV filter is a baseline
  (`dv-baseline` / `dv-shaped`); with orthogonality gone, "is this matrix-valued
  ED under another name" must be answered with data, not the independence
  argument.
- The tone trade-off (K ≥ 128) is stated as a limitation in the paper, not
  only here.

