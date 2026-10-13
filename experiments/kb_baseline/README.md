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

The two spatial metrics we have confirm this on the pre-re-run Kodak24 CSV
(fs-baseline vs kb-baseline, n=24; the current corpus has 124 images, so
these tables do not recompute from `results/kbbaseline.csv`):

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
- **DV's low-frequency excess is resolved; the filter is enabled**
  (`RUN_DV = True`). An earlier version of this bullet documented a 34× FS
  `lowfreq_frac` excess on kodim01 with the obvious causes (clipping in the
  design, opponent round-trip error, DC non-conservation of the taps) ruled
  out. The rewrite of `tramage.dv_filter` found and fixed three defects in
  the old design: it diffused in a **nonlinear opponent working space**, so
  Jensen bias leaked image-correlated low-frequency error into the metric
  space; its Yule-Walker assembly had a **wrong Kronecker ordering**; and
  its **iterated synthetic-training design was unstable**. The training-free
  closed-form design in Oklab conserves DC in the metric space by
  construction. S-CIELAB results are in the DV section below.

**Consequence:** any claim that rests on `noise_L` comparing FS against
KB/DV was provisional only until the re-run produced S-CIELAB ΔE, the one
metric that applies spatial CSF weighting and can actually see what an HVS
filter does. S-CIELAB is the arbiter; `noise_L` is not. The arbiter tables
below (the interaction table and the DV results section) supersede every
`noise_L` comparison in this README.

## Provisional observation, to be confirmed by S-CIELAB

At K=8 on `noise_L` (per-image means): `fs-shaped` 0.0416 vs `kb-baseline`
0.0908 and `kb-shaped` 0.0558. That is a large effect (124/124 images), and
54% is a lot of headroom, so the substitution story — *palette-geometry
shaping on a naive filter recovers the benefit of an HVS-tuned filter* — may
hold. But it could shrink or reverse once error is measured with spatial
weighting, because `noise_L` understates KB by construction. **Do not state
substitution as the headline until S-CIELAB confirms it.**

## KB's DC leak, and what shaping does to it — two regimes

The full four-cell `lowfreq_frac` table (pre-re-run Kodak24 corpus,
per-image means, n=24) splits into two regimes:

| K | kb-baseline | kb-shaped | fs-baseline | fs-shaped |
|---|---|---|---|---|
| 8 | 0.0323 | 0.0229 | 0.0328 | 0.0474 |
| 16 | 0.0161 | 0.0049 | 0.0074 | 0.0100 |
| 32 | 0.0114 | 0.0029 | 0.0048 | 0.0051 |
| 64 | 0.0108 | 0.0020 | 0.0033 | 0.0025 |

**K ≥ 16 — shaping repairs a genuine KB-specific excess.** The KB excess
(`kb-baseline − fs-baseline`) is +0.0067 to +0.0087, and after shaping the gap
(`kb-shaped − fs-shaped`) closes to −0.0005 to −0.0052 (KB at or below FS).
The interaction `(kb_b − fs_b) − (kb_s − fs_s)` is positive on 23–24/24 images.

**K = 8 — there is no excess to repair; the interaction is differential harm.**
The KB excess is −0.0005 (KB and FS start identical on lowfreq). The
interaction there comes from `fs-shaped` being 45% *worse* than `fs-baseline`
(0.0474 vs 0.0328) while `kb-shaped` improves (0.0229 vs 0.0323). So at K=8 the
mechanism is not "shaping repairs KB"; it is "shaping damages FS and does not
damage KB." Two different findings, reported separately.

Why shaping raises lowfreq on FS but not KB at small K: the shaping matrix
damps the palette's degenerate direction, and the redistributed error has to
go somewhere. KB's HVS taps have high-frequency emphasis, so they absorb it
into high-frequency structure (KB lowfreq *falls*). FS has no such emphasis,
so the redistributed error becomes low-frequency worms (FS lowfreq *rises*).
`spectral_anisotropy` rises for **both** arms (KB +0.67, FS +0.34 at K=8), so
the divergence is in frequency, not isotropy — both shaped arms get more
directional; they differ in whether that structure sits at low or high
frequency.

**For the paper — the K=8 worm cost, stated with its mechanism.** At K=8, on
the Kodak subset, shaping raises FS's low-frequency error by 52% (median of
per-image ratios, current 124-image corpus; pooled means give +41%), while
on CQ100 it lowers it by 32%; the full-corpus median is −23%. The sign
reversal is the convergence mechanism, not a caveat: shaping pulls
low-frequency error toward a common level, so images beginning below that
level rise and those beginning above it fall. (Supersedes the 45% figure
cited above — pooled means over the pre-re-run Kodak-only corpus.)

Median per-image change in `lowfreq_frac`, FS baseline → shaped:

| K | all 124 | Kodak 24 | CQ100 100 |
|---|---|---|---|
| 8 | −23% | +52% | −32% |
| 16 | −19% | −1% | −29% |
| 32 | −24% | −16% | −24% |
| 64 | −36% | −31% | −37% |
| 128 | −46% | −35% | −47% |
| 256 | −52% | −34% | −54% |

Read down the Kodak column as the mechanism weakening with K: the rise
shrinks through the K=16 null (−1%) into the general benefit as baseline
spread narrows.

(These are L-channel low-frequency fractions, not spatially CSF-weighted;
whether the same holds under S-CIELAB weighting remains open — the re-run
did not add such a metric.)

### The two-regime split is exploratory: robustness and caveats (30 Aug)

The 0.02 boundary was chosen by eye, after the re-run, and is not in the
preregistration. `analyze_regimes.py` (output: `results/regime_robustness.csv`)
reports the finding with its caveats rather than as an established result:

- **Threshold robustness.** The sign flip holds at every threshold from
  0.01 to 0.03 (figure window K=16–64: below-regime median excess
  +0.0056/+0.0051/+0.0048/+0.0046 at 0.01/0.015/0.02/0.03; above-regime
  −0.0020/−0.0031/−0.0062/−0.0125). The pattern is real, but the "above"
  magnitude triples across the range — the boundary is graded, not sharp.
- **The split confounds image with K.** The share of cells below 0.02 rises
  from 21% at K=8 to 98% at K≥128, so "the image's own low-frequency error"
  partly re-describes palette size. Per-K splits are in the CSV.
- **It is a low-frequency-energy statement, not a perceptual one.** On
  S-CIELAB, KB beats FS in only 16–29% of cells in either regime in the
  figure window (45% above the boundary at all-K) — there is no majority
  perceptual benefit in either regime.
- **Reconciliation.** ROADMAP §8 (19 Aug) quotes +0.0053/−0.0109; Figure 1
  shows +0.005/−0.006. The difference is provenance and window: §8 used the
  superseded pre-correction CSV; the figure computes on the current CSV over
  K=16–64 (+0.0048/−0.0062). The current CSV's all-K cut gives
  +0.0041/−0.0091. Both windows are printed by `analyze_regimes.py`.

For the paper: report the observation as exploratory, with the threshold
rule, its robustness range, and the per-K confound stated; do not frame it
as a perceptual benefit of the KB filter above the boundary.

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
  taps, designed per palette by a closed-form (training-free) Yule-Walker
  solve. The diffusion runs in Oklab, the metric space; the Poirson-Wandell
  opponent map enters the design only through its Jacobian at the palette
  centroid, the spatial CSFs are the tabulated Zhang-Wandell S-CIELAB
  kernels, and the DC constraint (taps summing to the identity) is imposed
  exactly via Lagrange multipliers. The exact prior art, not a special case.

KB still runs in the kolpatzik1992 opponent space. For DV the remaining
caveat is one of domain, not of approximation: the implementation is an
Oklab-domain reimplementation of DV&E, not a device-linear-RGB port of the
original pipeline. The metric caution applies to DV as to KB: evaluated
primarily on error variance, DV will be understated for the same reason.

## Disambiguation: this is not Damera-Venkata

Damera-Venkata & Evans' matrix is **optimised against an HVS colour-frequency
model**: it routes error into the frequencies and opponent channels the eye
is least sensitive to. The shaping matrix here comes from **palette
covariance**: it routes error into the directions the palette can actually
absorb. Different objects answering different questions, and they compose —
the `kb-shaped`/`dv-shaped` configs apply both. The `cell` variant is
additionally spatially adaptive where a DV&E matrix is fixed for the image.

## DV results: the full matrix filter under S-CIELAB

The 8 Sep re-run enabled the rewritten DV filter (`RUN_DV = True`), taking
the run to six configs and 4464 trials. `analyze_dv.py` recomputes the
tables below from `results/kbbaseline.csv`. Entries are median per-image %
S-CIELAB ΔE reductions with two-sided Wilcoxon signed-rank p-values;
positive means the first config of the pair has lower ΔE.

**Shaping on top of the DV filter** (dv-baseline → dv-shaped, pooled,
n = 124):

| K | median % ΔE reduction | Wilcoxon p |
|---|---|---|
| 8 | +1.0% | 8.6×10⁻⁴ |
| 16 | +1.4% | 0.044 |
| 32 | +1.5% | 0.28 |
| 64 | +2.4% | 0.12 |
| 128 | +1.5% | 0.75 |
| 256 | −0.3% | 0.62 |

**The same comparison, split by corpus:**

| K | Kodak (n=24) | CQ100 (n=100) |
|---|---|---|
| 8 | −1.2% (p = 0.029) | +1.3% (p = 3.8×10⁻⁶) |
| 16 | −7.0% (p = 0.0011) | +1.8% (p = 9.5×10⁻⁵) |
| 32 | −11.0% (p = 1.7×10⁻⁶) | +3.0% (p = 4.5×10⁻⁴) |
| 64 | −18.6% (p = 3.0×10⁻⁶) | +4.4% (p = 1.9×10⁻⁵) |
| 128 | −23.5% (p = 3.9×10⁻⁶) | +3.4% (p = 0.0022) |
| 256 | −30.6% (p = 1.2×10⁻⁷) | +2.7% (p = 0.0095) |

**Against the other filters** (pooled, n = 124):

| K | fs-shaped vs dv-baseline | dv-baseline vs fs-baseline | dv-baseline vs kb-baseline |
|---|---|---|---|
| 8 | +0.4% (p = 0.24) | +17.4% | +16.9% |
| 16 | +2.4% (p = 0.0044) | +21.7% | +21.8% |
| 32 | +3.1% (p = 0.028) | +18.3% | +24.2% |
| 64 | +3.2% (p = 0.028) | +9.8% | +24.0% |
| 128 | +2.0% (p = 0.27) | +6.8% | +23.2% |
| 256 | −1.3% (p = 0.75) | +3.6% | +22.1% |

(p-values in the last two columns: all ≤ 0.012 against fs-baseline, all
< 10⁻²⁰ against kb-baseline.)

Three conclusions:

1. **The DV filter is the strongest unshaped baseline.** Its median ΔE is
   the lowest of the three unshaped filters at every K (Table A of
   `analyze_dv.py`), 3.6–21.7% below fs-baseline and 16.9–24.2% below
   kb-baseline, with the margin over FS largest in the small-to-moderate-K
   regime of interest.
2. **The shaping × filter overlap is nearly complete for the full matrix
   filter.** Shaping adds 1–2% pooled on top of DV, and even that splits
   by corpus: a small but significant gain on CQ100 at every K, a growing
   harm on Kodak from K=16 (−7.0% to −30.6%). The matrix filter already
   routes error into the frequencies and opponent channels the eye is
   least sensitive to, which is most of what palette-geometry shaping
   provides on top of the weaker filters.
3. **Shaped Floyd-Steinberg matches or beats the DV baseline at K ≤ 64**
   (+0.4% at K=8, a null; +2.4% to +3.2% at K=16–64, p ≤ 0.028), so
   palette-geometry shaping delivers prior-art-level quality at one 3×3
   matrix multiply per pixel, without a per-palette filter design.

## Configurations

| Parameter | Value |
|---|---|
| Corpus | Kodak24 + CQ100 (124 images, 768×512 / 512×768) |
| Palette sizes | K ∈ {8, 16, 32, 64, 128, 256} |
| Seed | 0 |
| Configs per trial | 6 (FS/KB/DV × baseline/shaped; `RUN_DV = True`) |
| Total trials | 4464 (124 × 6 × 6) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Diffusion | FS, KB scalar, DV matrix — all serpentine |

## Reproducing

```bash
python experiments/kb_baseline/run_kbbaseline.py        # ~25 min
python experiments/kb_baseline/plot_kbbaseline.py       # metrics + relative + interaction PNGs
python experiments/kb_baseline/analyze_interaction.py   # trend test + per-K TOST
python experiments/kb_baseline/analyze_dv.py            # DV tables (S-CIELAB, Wilcoxon)
```

## Verdict

**Shaping helps on top of every filter, and its benefit is largest where the
filter is weakest. The interaction is monotonic in K (Jonckheere–Terpstra
p = 2×10⁻²⁷, n = 124). On S-CIELAB — the arbiter metric — the interaction is
positive where the shaping effect is non-negligible (K ≤ 32, CIs clear of
zero) and declines monotonically (JT p = 2×10⁻¹⁹); from K = 64 the CIs
straddle zero as the main effect collapses. Shaping helps the weaker FS
filter more in the regime that matters, and the `noise_L` crossover below
does not survive spatial weighting there.**

### The interaction term

Reported two ways, with the **trend test as primary** (a single
degree-of-freedom test avoids the multiplicity of six per-K tests):

- **Primary — trend test.** Jonckheere–Terpstra ordered trend across K:
  **p = 2×10⁻²⁷** (n = 124; mean per-image slope −0.0021 per doubling,
  reported as a descriptive effect size only — a linear-in-log₂K fit is a
  poor model of the plateauing interaction). The interaction declines
  monotonically with K.
- **Secondary — per-K TOST** against a pre-specified margin θ = 25% of the
  pooled shaping benefit (90% bootstrap CI, α = 0.05 one-sided). Equivalence
  holds at K=16–64 and fails at K=8, 128 and 256.

L-channel noise, per-image (n = 124), mean [90% CI]:

| K | main effect | interaction | ratio (interaction / main) | verdict |
|---|---|---|---|---|
| 8 | 0.0356 | +0.0085 [+0.0065, +0.0106] | +0.24 [+0.18, +0.30] | not equivalent — FS benefits more |
| 16 | 0.0306 | +0.0043 [+0.0027, +0.0060] | +0.14 [+0.09, +0.20] | equivalent |
| 32 | 0.0199 | +0.0001 [−0.0008, +0.0009] | +0.00 [−0.04, +0.05] | equivalent |
| 64 | 0.0119 | −0.0021 [−0.0026, −0.0015] | −0.17 [−0.22, −0.13] | equivalent |
| 128 | 0.0068 | −0.0021 [−0.0024, −0.0017] | −0.31 [−0.36, −0.25] | not equivalent — KB benefits more |
| 256 | 0.0037 | −0.0018 [−0.0019, −0.0016] | −0.48 [−0.53, −0.42] | not equivalent — KB benefits more |

The **absolute interaction crosses zero between K=32 and K=64**, inside the
mid-range equivalence band (the quantity the per-K table reports). The trend
test is the single-DOF result for the monotonic decline; it does not report
a fitted crossing, because a linear-in-log₂K fit is a poor model of the
plateauing interaction.

Three things follow:

1. **The interaction is monotonic in K** (Jonckheere–Terpstra p = 2×10⁻²⁷)
   and crosses zero in the mid-range: shaping helps the weaker FS filter more
   at small K, the KB filter more at large K. Largest where palette geometry
   is most irregular — a mechanistic finding.
2. **At K=8 the interaction is 24% [18%, 30%] of the shaping effect** — the
   same order as the effect. Whether this is substitution or a metric
   artifact is answered by S-CIELAB below.
3. **The large-K negative ratio is a small-denominator artifact.** At K=256
   the main effect is 0.0037 and the interaction −0.0018; both are tiny in
   absolute terms. Absolute interaction is reported alongside the ratio so the
   ratio is not misread as a large effect.

Tone-error ratios are omitted from the headline because the tone main effect
crosses zero at K≥128 (the known tone-accuracy trade-off, reported explicitly
as a limitation).

S-CIELAB ΔE, per-image (n = 124), mean [90% CI] — the arbiter metric,
computed with the tabulated Zhang & Wandell kernels (30 Aug re-run; the
previous table was the Gaussian-approximation run):

| K | main effect | interaction | ratio (interaction / main) | verdict |
|---|---|---|---|---|
| 8 | 2.288 | +0.856 [+0.584, +1.141] | +0.37 [+0.25, +0.50] | FS benefits more |
| 16 | 1.736 | +0.611 [+0.417, +0.807] | +0.35 [+0.24, +0.46] | FS benefits more |
| 32 | 0.856 | +0.293 [+0.189, +0.400] | +0.34 [+0.22, +0.46] | FS benefits more |
| 64 | 0.381 | +0.053 [−0.005, +0.114] | +0.14 [−0.01, +0.30] | positive, CI straddles zero |
| 128 | 0.141 | +0.005 [−0.030, +0.041] | +0.03 [−0.22, +0.30] | null (main effect small) |
| 256 | 0.032 | −0.006 [−0.027, +0.017] | −0.18 [−1.11, +0.66] | null (small denominator) |

The interaction is positive and clear of zero where the shaping effect is
non-negligible (K ≤ 32) and declines monotonically (JT p = 2×10⁻¹⁹). From
K = 64 the CIs straddle zero as the main effect collapses; at K = 256 the
point estimate is slightly negative, nonsignificant. The `noise_L` sign
change above is the metric artifact in the regime that matters; "positive
at every K with confidence" no longer holds at large K, and is not claimed.

### What this changes for the paper

- The abstract states the regime, the mechanism, and the factorial design
  (what palette geometry predicts), and lets the February manuscript carry
  whichever headline the S-CIELAB data supports. Substitution is not claimed
  on `noise_L` alone.
- The shaping × filter interaction is reported as a **single trend test**
  (Jonckheere–Terpstra; p = 2×10⁻²⁷ on `noise_L`, p = 2×10⁻¹⁹ on S-CIELAB
  under the corrected kernels), with per-K TOSTs secondary. On S-CIELAB the
  interaction is positive with clear CIs at K ≤ 32 and fades with the main
  effect; no large-K sign change is claimed.
- **DV&E is now enabled** (`RUN_DV = True`) after the closed-form redesign,
  and it changes the baseline story: the full matrix filter is the strongest
  unshaped baseline, shaping adds nothing reliable on top of it (a small
  CQ100-only gain, harmful on Kodak from K=16), and shaped FS matches or
  beats the DV baseline at K ≤ 64 (see the DV results section). The
  metric-blindness caveat applies to DV for the same reason it applies to
  KB.
- The tone trade-off (K ≥ 128) is stated as a limitation in the paper, not
  only here.

