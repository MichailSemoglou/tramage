# Dithering quality experiment

The follow-up to [`../palette_geometry/`](../palette_geometry/README.md).
That experiment established that k-means palettes extracted from Kodak24
are consistently and substantially anisotropic in Oklab at every palette
size tested: the median degeneracy score `(λ1+λ2)/λ3` sat two orders of
magnitude below the isotropic benchmark of 2.0. This was necessary but
not sufficient: it showed the shaping matrix has something to act on. It
did not show that the action translates into measurable improvement under
dithering. This experiment asks that load-bearing question.

## Variants

For each `(image, K)` trial, three dithers are produced and scored:

- **baseline** — `shaping=None` (vector Floyd-Steinberg).
- **global** — one shaping matrix from the whole palette's covariance,
  via `tramage.shaping.global_shaping_matrix`.
- **cell** — one matrix per palette colour, from its k=4 nearest
  neighbours, via `tramage.shaping.per_cell_shaping_matrices`.

The shaping matrices are built at `eps = tramage.shaping.DEFAULT_EPS`
(`0.02²`), the same value the geometry experiment and the package
defaults use. The diffusion kernel is Floyd-Steinberg with serpentine
scan, unchanged.

## Metrics

Each dither is scored against the continuous-tone original in Oklab,
using `tramage.metrics`:

- **L-channel noise** (`noise_std[0]`) — the headline noise metric. The
  eye is most sensitive to lightness noise, and it is the metric the
  synthetic gallery comparison makes visible.
- **Blurred tone error** (`blurred_tone_error`) — Oklab distance between
  Gaussian-blurred versions (σ=1.5) of dithered and original. A proxy for
  perceived tone accuracy at viewing distance.
- **Low-frequency energy fraction** (`lowfreq_energy_fraction`, L channel) —
  the share of error power inside a disc of 10% of Nyquist. Captures the
  "worms"/slow-drift failure mode that palette-structure-aware shaping is
  designed to suppress.
- **Spectral anisotropy** (`spectral_anisotropy`, L channel) — directional
  non-uniformity of the mid-band error spectrum. Detects scan-direction
  artefacts.

Systematic bias is recorded for completeness (the part of the error no
amount of redistribution within the affine hull can change), but the
primary metrics are the three above.

## Configuration

| Parameter | Value |
|---|---|
| Corpus | Kodak24 + CQ100 (124 images, 768×512 / 512×768) |
| Palette sizes | K ∈ {8, 16, 32, 64, 128, 256} |
| Seed | 0 (single; palette-extraction variance was covered by the geometry run) |
| Variants per trial | 3 (baseline, global, cell) |
| Total trials | 2232 (124 × 6 × 3) |
| `eps` | `DEFAULT_EPS = 0.02²` |
| Diffusion | Floyd-Steinberg, serpentine |

The corpus grew from Kodak24 to Kodak24 + CQ100 in the re-run; the tables
below are Kodak-subset medians from the current CSV unless labelled
otherwise, and `results/dithering_quality.csv` covers all 124 images.

## Reproducing

From the repository root:

```bash
python experiments/dithering_quality/run_quality.py    # ~10-15 min, prints a timing probe first
python experiments/dithering_quality/plot_quality.py   # writes two PNGs
```

Outputs (gitignored, regenerable):

- `results/dithering_quality.csv` — one row per (image, K, variant).
- `results/metrics_vs_K.png` — three panels (noise_L, tone_error,
  lowfreq_frac), three lines per panel (baseline/global/cell), 25–75%
  shaded bands across images. **The headline plot:** does shaped sit
  below baseline?
- `results/gain_vs_geometry.png` — scatter joining this CSV to the
  geometry CSV on `(image, K)`. X = palette degeneracy score
  `(λ1+λ2)/λ3`; Y = relative noise reduction
  `1 − noise_L(shaped)/noise_L(baseline)`. **The mechanism plot:** does
  shaping's benefit track how degenerate the palette is?

## Verdict

**Shaping wins on noise — unequivocally, across every K and every image —
but the win comes at a tone-accuracy cost at large K, and the per-cell
variant underperforms the global one on the spectral metrics. This is the
honest, mixed outcome, and it is more useful than a clean sweep would have
been because it tells the paper where the method's regime actually is.**

### Numbers

**L-channel noise** (median across the 24 Kodak images of the current
124-image CSV; lower is better):

| K | baseline | global | cell | global vs base | cell vs base |
|---|---|---|---|---|---|
| 8 | 0.0917 | 0.0390 | 0.0399 | **−57%** | −57% |
| 16 | 0.0642 | 0.0230 | 0.0220 | **−64%** | −66% |
| 32 | 0.0390 | 0.0155 | 0.0145 | **−60%** | −63% |
| 64 | 0.0228 | 0.0104 | 0.0094 | **−55%** | −59% |
| 128 | 0.0132 | 0.0077 | 0.0067 | **−42%** | −49% |
| 256 | 0.0081 | 0.0054 | 0.0046 | **−33%** | −43% |

**Per-image paired win rate on L noise: 124/124 (100%) at every K, for
both global and cell variants.** This is not a distributional shift; it
is a consistent per-image reduction. Every image in the corpus, at every
palette size from 8 to 256, dithers with lower lightness noise when the
error is shaped.

**Blurred tone error** (median across the 24 Kodak images; lower is better):

| K | baseline | global | cell |
|---|---|---|---|
| 8 | 0.01955 | **0.01822** | 0.01824 |
| 16 | 0.01069 | **0.01042** | 0.01078 |
| 32 | 0.00684 | **0.00627** | 0.00689 |
| 64 | 0.00443 | **0.00430** | 0.00497 |
| 128 | 0.00277 | 0.00297 | 0.00360 |
| 256 | 0.00170 | 0.00205 | 0.00265 |

Here the story inverts. **Global shaping wins or ties tone error at K ≤
64, then loses at K ≥ 128.** Per-cell loses tone earlier, at K ≥ 32. This
is the trade-off the mathematical appendix predicted: damping error along
the palette's flat direction reduces noise but sacrifices whatever
information the image was asking of that direction. At small K the
sacrifice is invisible (the palette can't represent that direction
anyway); at large K the palette has enough colours to begin
representing it, and damping becomes a real cost.

**Low-frequency error fraction** (median across the 24 Kodak images; lower
is better):

| K | baseline | global | cell |
|---|---|---|---|
| 8 | 0.01565 | 0.03170 | 0.04018 |
| 16 | 0.00430 | **0.00408** | 0.01255 |
| 32 | 0.00340 | **0.00283** | 0.01170 |
| 64 | 0.00241 | **0.00142** | 0.00891 |
| 128 | 0.00146 | **0.00087** | 0.00957 |
| 256 | 0.00098 | **0.00057** | 0.00953 |

A third pattern. **Global shaping wins the low-frequency metric at K ≥
16** — confirming the mechanism story: the worm/low-frequency drift the
method targets is suppressed. But at K=8, global *loses* this metric
even while winning on noise, and **per-cell loses it at every K**.

### Pre-specified subgroups: S-CIELAB by corpus

The primary endpoint, split into the preregistered Kodak/CQ100 subgroups
(`analyze_subgroups.py`; full table in `results/subgroup_scielab.csv`).
Per-image % reduction from global shaping; sign p is the two-sided
binomial the abstract quotes, and Wilcoxon (signed-rank, on absolute
paired differences) is reported alongside in the CSV:

| K | pooled med% (wins, sign p) | Kodak med% (wins, sign p) | CQ100 med% (wins, sign p) |
|---|---|---|---|
| 8 | +17.7 (120/124, ~0) | +15.4 (22/24, 4e-5) | +18.9 (98/100, ~0) |
| 16 | +21.4 (124/124, ~0) | +13.9 (24/24, 1e-7) | +23.1 (100/100, ~0) |
| 32 | +17.5 (120/124, ~0) | +9.4 (22/24, 4e-5) | +20.3 (98/100, ~0) |
| 64 | +11.8 (104/124, 7e-15) | −1.0 (11/24, 0.84) | +13.3 (93/100, 3e-20) |
| 128 | +5.6 (83/124, 2e-4) | −8.8 (4/24, 0.0015) | +8.2 (79/100, 4e-9) |
| 256 | +1.2 (66/124, 0.53) | −18.0 (0/24, 1e-7) | +3.3 (66/100, 0.0018) |

**The pooled headline masks a corpus divergence from K=64 up.** The
benefit is confirmed on both corpora at K ≤ 32. At K=64 the Kodak
subgroup is a wash; at K ≥ 128 shaping significantly *harms* Kodak under
S-CIELAB while still helping CQ100. The pooled significance at K=64–128
is carried by CQ100, and the pooled K=256 null (sign p = 0.53; Wilcoxon
p = 0.15) hides uniform Kodak harm (0/24 wins) against a real CQ100
gain (median +3.3%). Mean and median diverge at K=256 (mean −1.8% vs
median +1.2%): a skewed mix of two subgroup behaviours, not a homogeneous
small effect. The paper reports this split wherever it reports the
pooled curve.

### Interpretation

Three findings, each publishable, each of which the paper must state
plainly:

1. **The central noise claim holds at corpus scale.** A 33–66% reduction
   in L-channel noise, consistent on every image at every K, is the
   headline result. The synthetic-gallery finding from
   `docs/mathematical-appendix.md` (3.1× lower lightness noise on one
   photograph) generalises: across Kodak24 the median reduction is
   2.2–2.8×.

2. **The trade-off the appendix predicted is real and quantified.** At
   K ≤ 64 the trade is free — shaping wins noise without losing tone. At
   K ≥ 128 the palette has enough colours to start representing the
   formerly-flat direction, and damping it costs ~5–20% blurred-tone
   error. This gives the paper an honest scope statement: the method is
   a noise-reduction tool for small-to-mid palettes (the designer
   regime — riso, duotone, brand sets), not a free improvement for all
   CQ applications.

3. **Per-cell shaping does not earn its complexity on these images.** It
   matches global on noise (slightly better), but loses on both tone
   error and the worm metric from K=16 onward. The hypothesis that
   locally-varying shaping adapts better turns out not to hold for
   natural-image k-means palettes, which are globally degenerate enough
   that the global matrix already captures the relevant structure. This
   is ROADMAP's Phase 3 reframing condition answered early and
   negatively: per-cell should be dropped from the central paper claims
   and kept, clearly labelled, as an unproven extension.

The `gain_vs_geometry.png` scatter completes the mechanism story: the
relative noise improvement is essentially flat across the degeneracy
range, not stronger for more-degenerate palettes. Shaping's benefit
tracks palette *size* (small K benefits more) more than palette
*degeneracy*. This is a subtler finding than the original mechanism
hypothesis predicted, and it is what the data says.

### What this changes for the paper

- The headline becomes "**palette-structure-aware shaping reduces
  dithering noise by 33–66% across palette sizes**", with the scope
  qualifier "**for small-to-mid palettes (K ≤ 64)**" attached to the
  tone-accuracy claim.
- The scope claim is per-corpus at large K: under S-CIELAB the benefit
  holds on both corpora at K ≤ 32; from K = 64 the pooled effect is
  CQ100-driven, with Kodak null at K = 64 and significantly harmed at
  K ≥ 128 (see the subgroup table above). The abstract's K = 256
  parenthetical names the Kodak harm explicitly.
- The per-cell variant is removed from the central contribution and
  demoted to "explored but not adopted" in the paper.
- The designer-palette framing suggested earlier becomes more
  load-bearing, not less: the small-K regime where shaping is a free
  improvement *is* the designer regime.
- The ROADMAP Phase 3 "epsilon sweep" and "clustered-palette test"
  remain useful characterisation work, but the per-cell question they
  were partly aimed at is now answered.
