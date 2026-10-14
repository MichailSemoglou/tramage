# Epsilon sweep

## Purpose

The epsilon parameter (`eps`) in `tramage.shaping` controls how aggressively
error is damped along near-degenerate palette axes. This sweep varies it
across orders of magnitude for a fixed image and palette, producing the
noise-vs-tone Pareto front that characterizes the trade-off.

## What it tests

For each eps on a log-spaced grid from 1e-6 to 1e-1:

- **noise_std (L channel)**: how much lightness noise remains in the
  dithered output.
- **blurred_tone_error**: how closely the blurred dithered output matches
  the blurred original (a proxy for tone reproduction).
- **S-CIELAB ΔE**: the field's standard perceptual metric for halftoning
  (Zhang & Wandell 1997), which accounts for the contrast sensitivity
  function.

## Running it

```bash
python experiments/epsilon_sweep/run_sweep.py              # synthetic image, K=8
python experiments/epsilon_sweep/run_sweep.py --K 16       # larger palette
python experiments/epsilon_sweep/run_sweep.py --image photo.jpg  # real image
```

Then plot:

```bash
python experiments/epsilon_sweep/plot_sweep.py
```

## What to look for

- **Pareto front** (`results/sweep_pareto.png`): each point is one eps
  value. The curve traces the trade-off between noise and tone; eps values
  near the knee offer the best compromise.
- **S-CIELAB curve** (`results/sweep_scielab.png`): ΔE as a function of eps.
  A rising curve means larger eps hurts perceptual quality despite lower
  noise — suggesting an optimal eps near the minimum or at a specific
  transition.
- The default eps (`DEFAULT_EPS = 0.02 ** 2 ≈ 4e-4`) is marked on both
  plots; the sweep should show how close it sits to any empirically optimal
  knee.

## Interpretation

This sweep is the empirical backbone of the paper's characterization
section. Running it across multiple palettes and images will answer:

1. Does the optimal eps (Pareto knee) correlate with a palette's eigenvalue
   scale? If so, eps becomes derivable rather than tunable.
2. Is the default eps conservative, aggressive, or about right on real
   data?
3. Does S-CIELAB agree with the project's own metrics about where the best
   trade sits?

## Corpus-scale sensitivity (30 Aug)

`run_corpus_sweep.py` answers question 2 at corpus scale: Kodak24 x
K ∈ {8, 16, 32} x eps ∈ {0.25x, 0.5x, 1x, 2x, 4x} the default
(`results/corpus_sweep.csv`, 432 trials; baseline dither shared per
(image, K)). Median per-image % change vs baseline:

| K | metric | 0.25x | 0.5x | 1x | 2x | 4x |
|---|---|---|---|---|---|---|
| 8 | S-CIELAB ΔE | +16.4 | +16.0 | +15.4 | +14.1 | +13.0 |
| 8 | noise_L | +47.1 | +50.8 | +53.6 | +55.4 | +56.5 |
| 8 | tone_error | +10.3 | +10.0 | +9.3 | +9.0 | +8.1 |
| 16 | S-CIELAB ΔE | +19.9 | +17.1 | +13.9 | +10.6 | +9.3 |
| 16 | noise_L | +53.7 | +56.3 | +58.7 | +61.1 | +62.7 |
| 16 | tone_error | +11.6 | +9.4 | +6.4 | +1.7 | −0.7 |
| 32 | S-CIELAB ΔE | +17.4 | +13.6 | +9.4 | +5.6 | +1.6 |
| 32 | tone_error | +14.2 | +9.0 | +4.2 | −0.6 | −4.4 |

S-CIELAB win rates stay high at K=8–16 across the range (worst 21/24),
softening only at K=32 with aggressive damping (14/24 at 4x).

Findings:

1. **The headline is qualitatively robust, quantitatively eps-sensitive.**
   The S-CIELAB benefit never changes sign within the tested range, but its
   magnitude at K=16 spans 9–20% across a 16x eps range. The paper states
   the default's neighborhood, not just the default.
2. **The default is conservative on Kodak.** Smaller eps gives a *larger*
   S-CIELAB benefit at every K tested, trading away noise reduction for
   tone (noise_L reduction grows with eps while tone_error shrinks and then
   turns negative — the fidelity-noise dial behaving as the appendix
   predicts). Whether 0.25x-0.5x is the better default is a real question
   the February manuscript should address, ideally on both corpora.
3. **eps is an absolute scale, and it shows.** Because the default is not
   normalized by palette extent, its meaning shifts with K; the corpus
   table is the evidence for the disclosure sentence in the paper.
