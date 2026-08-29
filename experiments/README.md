# Experiments

This folder holds evaluation that goes beyond the synthetic cases in
`gallery/`. Each section names its own status and reproduction path.

## Palette geometry check (done)

`palette_geometry/` answers the Phase 1 reframing question — *are k-means
palettes extracted from standard image corpora consistently near-degenerate
in Oklab, or is near-degeneracy the exception?* — by running
`tramage.palette.eigenspectrum_report` across Kodak24 at
K ∈ {8, 16, 32, 64, 128, 256} with three seeds each. No new method code:
the harness exercises the package's public API unchanged. See
[`palette_geometry/README.md`](palette_geometry/README.md) for the verdict
and reproduction instructions.

## Dithering-quality run (done)

`dithering_quality/` runs the three variants (vector Floyd-Steinberg,
global-covariance shaping, per-cell shaping) across Kodak24 and CQ100
(124 images) at K ∈ {8, 16, 32, 64, 128, 256} and scores each on the metrics
in `tramage.metrics`: systematic bias, blurred tone error, per-channel
noise, low-frequency energy fraction, spectral anisotropy, and S-CIELAB ΔE.
The result is a 2232-row CSV and two figures: a metrics-by-K panel plot and
a gain-vs-geometry scatter that joins this experiment to the palette
geometry check. See [`dithering_quality/README.md`](dithering_quality/README.md)
for the verdict. In short: shaping reduces L-channel noise by a median
32–59% (Kodak) / 21–41% (CQ100) across K, winning on every image at every
K; a tone-accuracy trade-off appears only at K ≥ 128 on Kodak and not at
all on CQ100; per-cell does not earn its complexity and is dropped from
the paper's central claims.

## Epsilon sweep (done)

`epsilon_sweep/` varies `eps` in `tramage.shaping` across several orders
of magnitude for a fixed palette and image, producing a noise-vs-tone
Pareto front and an S-CIELAB ΔE curve.

For a fixed palette and image, vary `eps` in `tramage.shaping` across several
orders of magnitude and plot the fidelity-versus-noise trade-off it
produces (recovered off-plane tone against lightness noise, as in the
near-degenerate gallery comparison). The purpose is to show `eps` as a
continuous dial rather than a value to be tuned once and forgotten, and to
give a starting point for choosing it relative to a palette’s own
eigenvalue scale.

## Clustered-palette test (closed, negative)

Originally planned to confirm that per-cell shaping adapts to palettes
with multiple distinct clusters of near-degeneracy, where a single global
matrix would average across them. Closed by the
[`dithering_quality/`](dithering_quality/README.md) corpus run: on the
124-image corpus, per-cell holds up on noise but loses on tone accuracy and
the worm metric from K=16 onward. Natural-image palettes are
globally degenerate enough that the global matrix already captures the
relevant structure. Per-cell remains in the package as an explored
extension; this experiment is no longer motivated by the paper's claims.

## Colour-space comparison (done)

`colour_space/` tests whether Oklab — the package default — is actually
the right colour space for palette-structure-aware shaping, or whether
CIELAB gives shaping more purchase. Adds a `srgb_to_cielab` /
`cielab_to_srgb` path to `tramage.colour` (D65) for the comparison, then
runs the same Kodak24 harness with shaping in each space, each compared
against its own baseline. See
[`colour_space/README.md`](colour_space/README.md) for the verdict. In
short: Oklab gives shaping roughly double the noise-reduction purchase
of CIELAB (52–59% vs 12–31% at K=8–64) and wins on 23/24 images at every
K — the package default is corroborated by corpus evidence.

## CQ100 dithering-quality run (done)

`cq100_quality/` replicates the Kodak dithering-quality experiment on the
CQ100 corpus (Celebi & Perez-Delgado 2023, JEI 32(3), 033019): 100 images
across 8 categories (animals, food, objects, people, places, plants,
vehicles, miscellaneous), CC BY 4.0, all 768×512 or 512×768 PNG. Dataset
must be downloaded manually from https://data.mendeley.com/datasets/vw5ys9hfxw
and extracted to `experiments/corpora/cq100/`. Then:

    python experiments/cq100_quality/fetch_cq100.py   # confirms images present
    python experiments/cq100_quality/run_cq100.py     # 1800-trial corpus run
    python experiments/cq100_quality/plot_cq100.py    # metrics-vs-K + gain-vs-geometry

Results (27 July 2026): global shaping reduces L-channel noise by 22–41%
across K, winning on 100/100 images at every K. Unlike the Kodak24 run, no
tone-accuracy trade-off appears at any K — global beats baseline on blurred
tone error at every palette size. Low-frequency (worm) energy is halved.
Per-cell beats global on raw noise at K ≥ 16 but loses on tone and worms,
strengthening the Kodak conclusion that per-cell does not earn its
complexity. The 100% win rate on a 100-image corpus with category diversity
substantially strengthens the paper's central claim relative to the
single-corpus Kodak24 result.

## Damera-Venkata baseline (done, scalar KB scope)

`kb_baseline/` tests whether palette-structure-aware shaping still helps
when the baseline diffusion filter is HVS-optimised rather than plain
Floyd-Steinberg. Uses the scalar Kolpatzik-Bouman filter
(kolpatzik1992 Table 1) — the separable special case of Damera-Venkata
and Evans' matrix-valued filter — as the stronger baseline, with four
configurations (FS baseline, FS+shaped, KB baseline, KB+shaped). See
[`kb_baseline/README.md`](kb_baseline/README.md) for the verdict. In
short: shaping reduces noise by a median 22–33% on top of the KB filter,
winning on 124/124 images at every K — the generalisation claim holds, at
reduced magnitude relative to the FS comparison. The full matrix-valued DV filter
remains as a possible follow-up if a reviewer demands the exact
comparison; the scalar KB result is indicative.
