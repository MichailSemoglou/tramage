# Pre-registration of the 2026 re-run analyses

**Frozen 30 August 2026.** This file now separates what was written before
the re-run produced data from what was added after, so that the paper's
"pre-registered" claims are exactly checkable against git history. Nothing
has been deleted; sections have been moved, not rewritten.

Timeline (all 19 August 2026, UTC+8):

- 00:31 (`3e4ca54`) — file created: primary endpoint, hypotheses,
  equivalence margin, reporting conventions, first-analysis plan,
  provenance.
- 00:40 (`3d9aee1`) and 00:49 (`dd4eaba`) — hypothesis 3 reworded
  (substitution made provisional on S-CIELAB; Jonckheere-Terpstra trend
  test named). Still before any data.
- ~01:00 — the corpus re-run began (code commit `8931a13`); the CSVs were
  finalised at 16:56 and 17:33.
- 12:39 (`40720f1`) — no content change to this file's claims.
- 12:43 (`b8e802d`) — **Amendment A** (K=8 contingency), while the run was
  in flight, before its results were read.
- 17:16 (`5ec1205`) — rank-agreement results committed; data now read.
- 18:04 (`1556ef8`) — **Amendment B** (subgroup analysis), after the main
  CSVs.
- 18:12 (`647c90c`) — **Amendment C** (palette-geometry regression), after
  the main CSVs.
- 18:27–18:28 (`6a8dfd0`, `7b4d576`) — **Amendment D** (fixed-palette
  readout), after the main CSVs but before the fixed-palette run (20:19),
  so it is pre-registered *for that experiment*.

The term "pre-registered" is claimed only for Part 1. The amendments are
dated above and cited as amendments in the paper.

# Part 1 — pre-registered before the re-run

This file pre-specifies the primary endpoints, comparison family, equivalence
margin, and analysis procedures for the four dithering-quality re-runs
(Kodak24, CQ100, KB factorial, Oklab-vs-CIELAB). It exists so the statistical
claims in the manuscript are declared *before* the data that produces them is
collected, not reverse-engineered afterwards.

## Primary endpoint and family

- **Primary metric per hypothesis:** S-CIELAB ΔE (`scielab_de`), the field's
  standard perceptual ruler (Zhang & Wandell 1997). It carries the claim.
- **Mechanism metrics:** `noise_L`, `tone_error`, `lowfreq_frac`,
  `spectral_anisotropy` explain *why*; reported as secondary and descriptive,
  without multiplicity-correction claims.
- **Family size:** one primary metric per hypothesis. The mechanism metrics are
  not corrected; they are described as diagnostics.

## Hypotheses

1. **Main.** Global palette-structure-aware shaping reduces S-CIELAB ΔE
   relative to unshaped vector Floyd-Steinberg, on Kodak24 and on CQ100.
2. **Mechanism.** The benefit is driven by reduced low-frequency ("worm")
   error, tracked by `lowfreq_frac` and `spectral_anisotropy`.
3. **Interaction + trend (the FS/KB factorial).** The primary test is a
   single-degree-of-freedom **trend test**: the per-image slope of the FS−KB
   interaction against log₂K, tested by a one-sample t-test on the 24 per-image
   slopes. Per-K TOST equivalence tests (θ = 25% of the pooled shaping benefit)
   are secondary, used only to locate the sign change. Whether the small-K
   asymmetry implies *substitution* (shaping on a naive filter recovers the
   benefit of an HVS-tuned filter) is tested by S-CIELAB ΔE, not by `noise_L`:
   `noise_L` is spatially unweighted and blind to the filters' energy-
   redistribution mechanism.

4. **DV (exact prior art).** Shaping still reduces S-CIELAB ΔE on top of the
   full matrix-valued Damera-Venkata filter (`dv-shaped` vs `dv-baseline`).

## Equivalence margin (pre-specified)

The shaping × filter interaction is deemed negligible if its 90% bootstrap CI
lies within ±θ, where **θ = 0.25 × the pooled shaping benefit** (the main
effect of shaping) at that K. Equivalence is assessed with a two-one-sided
test (TOST) at α = 0.05 one-sided (10,000 bootstrap resamples, seed 0).
This is the **secondary** analysis; the primary interaction analysis is the
single-DOF trend test in hypothesis 3. Both are implemented in
`experiments/kb_baseline/analyze_interaction.py`.

## Reporting conventions

- Per-image paired differences with effect sizes and bootstrap confidence
  intervals, not p-values alone. No bare `p < 0.05` without an interval.
- The term "pre-registered" is claimed only for the analyses declared in
  Part 1 of this file, and only because Part 1 predates the re-run. The
  earlier ROADMAP median-table figures are exploratory and are not described
  as pre-registered.

## First analysis after the re-run

Before writing anything, check whether `scielab_de` and `noise_L` rank the
conditions the same way: per (image, K), which variant wins, plus Spearman ρ
on the paired values. Agreement gives the planned paper (S-CIELAB carries the
claim, Oklab metrics are the mechanism). Disagreement changes the paper's
structure and is the more interesting result; it must be known first.
Implemented in `experiments/analyze_rank_agreement.py`.

## Provenance

Every result CSV records `git_sha` (repo commit, suffixed `-dirty` if the tree
is uncommitted) and `corpus_sha256` (SHA-256 over the sorted image paths and
each file's digest), so any figure can be traced to the exact code and corpus
that produced it.

# Amendments — added once the re-run was under way or complete

## Amendment A — contingent interpretation, the K=8 outcome (19 Aug, 12:43)

Added while the re-run was in flight (commit `b8e802d`), before its results
were read.

The small-K (K=8) shaping benefit currently rests on `noise_L`, which cannot
see spatial structure; both spatially aware metrics (`lowfreq_frac` +45%,
`spectral_anisotropy` +0.34) say shaping *hurts* at K=8. S-CIELAB is spatially
weighted, so the K=8 result is genuinely in question. Both outcomes are
pre-specified:

- **If S-CIELAB does not confirm shaping at K=8:** the paper relocates to a
  moderate-K intervention (K=16–64); K=8 becomes a boundary finding — below
  some palette size the redistributed error has nowhere to go that the eye
  does not see. The 45% worm cost is already named, so this is a finding, not
  a concession.
- **If S-CIELAB confirms shaping at K=8:** the perceptual gain from reduced
  colour error outweighs the added spatial structure — worth a paragraph, as
  that trade is the interesting claim.

The rank agreement (`scielab_de` vs `noise_L`) is the first result read from
the re-run; it decides which of the two papers is being written.

## Amendment B — subgroup analysis, Kodak vs CQ100 (19 Aug, 18:04)

Added after the main CSVs were finalised (commit `1556ef8`).

The two corpora are analysed jointly and as pre-specified subgroups (Kodak24,
24 images; CQ100, 100 images). Shaping is hypothesised to impose a
palette-determined low-frequency level that pulls both corpora toward a common
value, so the subgroup analyses are:

- S-CIELAB shaping benefit, per corpus.
- The lowfreq response: across-image variance collapse and the
  (global − baseline)-vs-baseline regression, per corpus.
- The KB DC-leak floor and the shaping-compensation interaction, per corpus.

This replaces any "independent replication" framing: the re-run analysed one
124-image set (a `corpus_dirs` bug combined the corpora), so the subsets are a
subgroup analysis, not independent confirmation.

## Amendment C — palette-geometry regression (19 Aug, 18:12)

Added after the main CSVs were finalised (commit `647c90c`).

The positive form of the "palette-determined" claim. **Pre-specified
predictor: condition number** of the palette covariance (λ_max/λ_min), the
scalar that directly expresses how badly conditioned a palette is — the
thesis's central quantity.

Test: regress post-shaping `lowfreq_frac` on log(condition number); then add
image index as a factor and compare R². If image identity contributes no
explanatory power beyond the condition number, the positive form holds:
post-shaping lowfreq is palette-determined, not image-determined. Eigenvalue
spread and nearest-neighbour spacing are reported as exploratory checks only,
not as pre-specified predictors.

## Amendment D — fixed-palette experiment readout (19 Aug, 18:27–18:28)

Added after the main CSVs (commits `6a8dfd0`, `7b4d576`) but before the
fixed-palette run (20:19), so it is pre-registered for that experiment.

A fixed palette is applied across many images to decouple palette geometry
from image content (the per-image-palette confound). Design: 10 seed palettes
spanning the K=8 condition-number range (3.1 to 1.6e5, including a badly
conditioned one), at K=8 and K=16, each applied to all 124 corpus images with
global shaping; post-shaping `lowfreq_frac` is recorded.

**Readout: the intraclass correlation** ICC(1) = between-palette variance /
total variance (one-way random-effects, palettes as groups), computed
separately for K=8 and K=16. Pre-specified interpretation:

- **ICC > 0.5** — strong: the post-shaping level is largely palette-determined.
- **0.2 ≤ ICC ≤ 0.5** — partial: a palette component exists but is not dominant.
- **ICC < 0.2** — null: no meaningful palette-determined level.

**Mismatch confound (checked, not assumed).** With a fixed palette, images
differ in how well the palette fits them, so within-palette variance partly
reflects palette-image mismatch. If the within-palette variance is large, the
first alternative is mismatch; check it by testing whether high-variance images
under a palette are those with high baseline error under that palette.

**Readout refinement (pre-specified).** The condition numbers span five orders
of magnitude with a single outlier (yasaka_pagoda, cond 1.6e5; the next is
2.8e3). ICC is computed on the **nine well-conditioned palettes (cond < 3e3)**,
and the pagoda palette is reported separately as a case, so one point does not
carry the leverage on the pooled ICC. The case reading: the thesis predicts the
badly-conditioned palette produces the *most* palette-determined behaviour, i.e.
the lowest within-palette variance — the mechanism visible in a single case,
shown in a figure rather than an ICC number.

**Heteroscedasticity check.** ICC(1) assumes homoscedastic within-palette
variance. Check whether within-palette variance varies systematically with the
palette condition number; if well-conditioned palettes leave image content
intact while badly-conditioned ones flatten it, a single ICC averages two
regimes and describes neither, so report per-palette within variance against
condition number.
