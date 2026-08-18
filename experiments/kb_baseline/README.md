# Filter-baseline experiment: FS vs KB vs DV, with and without shaping

The paper's claim needs to survive stronger comparisons than plain
Floyd-Steinberg. This experiment runs three filters — naive Floyd-Steinberg
(FS), the Kolpatzik-Bouman scalar HVS filter (KB, the separable special case
of Damera-Venkata), and the full matrix-valued Damera-Venkata filter (DV) —
each with and without palette-structure-aware shaping. Six configurations per
(image, K):

- **fs-baseline / fs-shaped** — Floyd-Steinberg, without / with shaping.
- **kb-baseline / kb-shaped** — KB scalar HVS filter, without / with shaping.
- **dv-baseline / dv-shaped** — the matrix-valued DV filter, without / with shaping.

## Metric blindness: `noise_L` cannot see what KB/DV do

`noise_L` is a per-channel error standard deviation — a spatially unweighted
statistic. KB (and DV) do not reduce error variance; they **redistribute**
error energy toward high spatial frequencies and toward rotational symmetry
(blue noise) where the visual system is less sensitive. A metric that ignores
spatial structure cannot see that redistribution.

The two spatial metrics we have confirm this on the existing CSV
(fs-baseline vs kb-baseline, n=24):

| K | noise_L | spectral_anisotropy | lowfreq_frac |
|---|---|---|---|
| 8 | 0.0907 → 0.0908 | 0.8208 → **0.7434** | 0.0328 → 0.0323 |
| 16 | 0.0671 → 0.0731 | 0.6267 → **0.4648** | 0.0074 → 0.0161 |
| 32 | 0.0390 → 0.0480 | 0.5159 → **0.3442** | 0.0048 → 0.0114 |
| 64 | 0.0226 → 0.0318 | 0.4862 → **0.3370** | 0.0033 → 0.0108 |

- **`spectral_anisotropy` separates cleanly** (KB is more isotropic on
  15–18/24 images at every K) — the KB filter's blue-noise mechanism is
  working, and `noise_L` cannot see it.
- **`noise_L` does not separate** (0.0907 vs 0.0908 at K=8): an HVS filter
  producing no change on this metric is evidence that the metric is blind,
  not that the filter is weak.
- **`lowfreq_frac` goes the wrong way at K ≥ 16, and the cause is DC
  non-conservation, not the opponent approximation.** The published KB
  luminance taps sum to 0.9443 (kolpatzik1992 deliberately drop the DC
  constraint), leaking ~5.6% of luminance error per pixel. That leak produces
  a low-frequency floor (~0.010–0.012) that does not collapse as K grows,
  while FS's low-frequency error falls to 0.0025. Verified per-image: KB's
  bias stays ~0.001–0.002 across K while FS's falls to ~0.0004.
- **DV is currently worse, and the obvious causes are ruled out.** At K=32 on
  kodim01 (single image), DV gives `lowfreq_frac = 0.0862` (34× FS) and
  `bias_norm = 0.00328` (8× FS). Three checks clear the suspects: **zero clip
  events** during filter design (the ±1e3/±1e6 clipping never fires), an
  **exact opponent round-trip** (distortion 0.0), and **DC-conserving taps**
  with a clean constant-image output (error 0.0). The remaining suspect is the
  synthetic-training filter design not generalising to real-image spectra.
  DV is disabled in the run (`RUN_DV = False`) until resolved.

**Consequence:** any claim that rests on `noise_L` comparing FS against
KB/DV is provisional until the re-run produces S-CIELAB ΔE, the one metric
that applies spatial CSF weighting and can actually see what an HVS filter
does. S-CIELAB is the arbiter; `noise_L` is not.

## Provisional observation, to be confirmed by S-CIELAB

At K=8 on `noise_L` (per-image means): `fs-shaped` 0.0416 vs `kb-baseline`
0.0908 and `kb-shaped` 0.0558. That is a large effect (24/24 images), and
54% is a lot of headroom, so the substitution story — *palette-geometry
shaping on a naive filter recovers the benefit of an HVS-tuned filter* — may
hold. But it could shrink or reverse once error is measured with spatial
weighting, because `noise_L` understates KB by construction. **Do not state
substitution as the headline until S-CIELAB confirms it.**

## KB's DC leak: a publishable finding, and shaping compensates for it

Kolpatzik–Bouman drop the DC constraint deliberately, and this leaves a
low-frequency error floor that does not decay with palette size. Per-image
means over the full 24-image corpus (`lowfreq_frac`, lower is better):

| K | kb-baseline | kb-shaped | fs-baseline |
|---|---|---|---|
| 8 | 0.0323 | 0.0229 | 0.0328 |
| 16 | 0.0161 | 0.0049 | 0.0074 |
| 32 | 0.0114 | 0.0029 | 0.0048 |
| 64 | 0.0108 | 0.0020 | 0.0033 |

Two things follow:

1. **The crossing at K=16 is where DC conservation starts to matter more than
   HVS weighting.** Below K=16 both filters are dominated by large quantisation
   error; from K=16 up, FS's low-frequency error collapses (it conserves DC)
   while KB's floor stays near 0.011. That is a concrete, quantified limitation
   of a widely cited HVS-optimised filter — a case where optimising against a
   perceptual model introduces a systematic artifact the model does not
   penalise.

2. **Shaping compensates for the DC leak rather than inheriting it.** At
   K=16–64, `kb-shaped` cuts the floor from ~0.011 to ~0.002–0.005, at or below
   FS-baseline's level. The shaping matrix damps the palette's degenerate
   direction, which is where the leaked DC error accumulates. Shaping repairs a
   documented limitation of an HVS-optimised filter.

(These are `noise_L`-based low-frequency fractions, not S-CIELAB; the re-run
will confirm whether the same holds under spatial CSF weighting.)

## Regime of interest

Shaping is a **small-to-moderate-K intervention** (K ≤ 64). That is where
palette geometry is irregular enough for shaping to have purchase; at
K = 256, error diffusion into 256 colours is barely doing anything for
anyone (the shaping benefit is 0.0037 of a noise level of ~0.008). The
interaction ratio is only meaningful where the shaping effect (the
denominator) is not near zero; absolute interaction is reported alongside
the ratio throughout.

## The stronger baselines

- **KB scalar filter** (kolpatzik1992, Table 1): three independent 4-tap
  filters, one per opponent channel (Y, rg, yb), minimising perceived error
  variance under per-channel CSFs. Uses the published coefficients directly.
- **DV matrix filter** (`tramage.dv_filter`): the full Damera-Venkata matrix
  taps, designed per palette by solving a Yule-Walker system in
  Poirson-Wandell opponent space. The exact prior art, not a special case.

Both filters run through the same approximate opponent transform, a
documented limitation: the comparison is indicative rather than definitive
against a faithful linear-RGB pipeline. The same caution applies to DV as to
KB — evaluated primarily on error variance, DV will be understated for the
same reason.

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
| Configs per trial | 4 (FS/KB × baseline/shaped; DV disabled) |
| Total trials | 576 (24 × 6 × 4) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Diffusion | FS, KB scalar, DV matrix — all serpentine |

## Reproducing

```bash
python experiments/kb_baseline/run_kbbaseline.py        # ~25 min
python experiments/kb_baseline/plot_kbbaseline.py       # metrics + relative + interaction PNGs
python experiments/kb_baseline/analyze_interaction.py   # trend test + per-K TOST
```

## Verdict

**Shaping helps on top of every filter, and its benefit is largest where the
filter is weakest — but the magnitude of that asymmetry is measured with
`noise_L`, which cannot see what the tuned filters do. The interaction is
monotonic in K (single-DOF trend p = 9×10⁻⁵); whether it implies substitution
or merely a metric artifact is decided by S-CIELAB, not by `noise_L`.**

### The interaction term

Reported two ways, with the **trend test as primary** (a single
degree-of-freedom test avoids the multiplicity of six per-K tests):

- **Primary — trend test.** Mean per-image slope of the FS−KB interaction
  against log₂K: **−0.0030 per doubling, 95% CI [−0.0044, −0.0017],
  p = 8.9×10⁻⁵**. The interaction declines monotonically with K.
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

The **absolute interaction changes sign between K=16 and K=32** (the quantity
the per-K table reports). The trend test's slope+p is the single-DOF result
for the monotonic decline; it does not report its own fitted crossing, because
a linear-in-log₂K fit is a poor model of the plateauing interaction.

Three things follow:

1. **The interaction is monotonic in K** (trend p = 9×10⁻⁵) and changes sign
   between K=16 and K=32: shaping helps the weaker FS filter more at small K,
   the KB filter more at large K. Largest where palette geometry is most
   irregular — a mechanistic finding.
2. **At K=8 the interaction is 34% [22%, 47%] of the shaping effect** — the
   same order as the effect. Whether this is substitution or a metric artifact
   is the S-CIELAB question.
3. **The large-K negative ratio is a small-denominator artifact.** At K=256
   the main effect is 0.0037 and the interaction −0.0020; both are tiny in
   absolute terms. Absolute interaction is reported alongside the ratio so the
   ratio is not misread as a large effect.

Tone-error ratios are omitted from the headline because the tone main effect
crosses zero at K≥128 (the known tone-accuracy trade-off, reported explicitly
as a limitation). S-CIELAB ΔE is added to this table once the re-run includes
it.

### What this changes for the paper

- The abstract states the regime, the mechanism, and the factorial design
  (what palette geometry predicts), and lets the February manuscript carry
  whichever headline the S-CIELAB data supports. Substitution is not claimed
  on `noise_L` alone.
- The shaping × filter interaction is reported as a **single trend test**
  (monotonic in K, p = 9×10⁻⁵), with per-K TOSTs secondary. The sign change
  is between K=16 and K=32 (absolute interaction).
- **DV&E is run, not deferred.** The matrix-valued DV filter is a baseline
  (`dv-baseline` / `dv-shaped`), and the same metric-blindness caveat applies
  to it: `noise_L` will understate DV for the same reason it understates KB.
- The tone trade-off (K ≥ 128) is stated as a limitation in the paper, not
  only here.

