# Pre-registration of the 2026 re-run analyses

**Dated:** 19 August 2026 — before the corpus re-run that produces the
authoritative per-image data for the paper.

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
3. **Substitution + trend (the FS/KB factorial).** At small K, palette-
   geometry shaping on plain Floyd-Steinberg reaches (or beats) the
   HVS-optimised filters without shaping. The primary test is a single-degree-
   of-freedom **trend test**: the per-image slope of the FS−KB interaction
   against log₂K, tested by a one-sample t-test on the 24 per-image slopes.
   Per-K TOST equivalence tests (θ = 25% of the pooled shaping benefit) are
   secondary, used only to locate the crossing point.

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
- The term "pre-registered" is claimed only for the analyses declared in this
  file, and only because this file predates the re-run. The earlier ROADMAP
  median-table figures are exploratory and are not described as pre-registered.

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
