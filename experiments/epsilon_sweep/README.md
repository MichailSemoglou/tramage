# Epsilon sweep

## Purpose

The epsilon parameter (`eps`) in `tramage.shaping` controls how aggressively
error is damped along near-degenerate palette axes. This sweep varies it
across orders of magnitude for a fixed image and palette, producing the
noise-vs-tone Pareto front that characterises the trade-off.

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

This sweep is the empirical backbone of the paper's characterisation
section. Running it across multiple palettes and images will answer:

1. Does the optimal eps (Pareto knee) correlate with a palette's eigenvalue
   scale? If so, eps becomes derivable rather than tunable.
2. Is the default eps conservative, aggressive, or about right on real
   data?
3. Does S-CIELAB agree with the project's own metrics about where the best
   trade sits?
