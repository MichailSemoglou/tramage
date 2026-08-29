# Quickstart

## Installation

```bash
pip install tramage
```

Or, from a checkout of this repository:

```bash
pip install -e ".[dev]"
```

The base package depends on `numpy` and `scipy`. Loading photographs
requires the optional `image` extra (`pillow`):

```bash
pip install "tramage[image]"
```

## A minimal example

Tramage works in Oklab, so the first step is always to convert into it and
the last step is always to convert back.

```python
import numpy as np
from PIL import Image

from tramage import (
    srgb_to_oklab,
    oklab_to_srgb,
    extract_palette,
    global_shaping_matrix,
    diffuse,
)

# Load a photograph and convert to Oklab.
rgb = np.asarray(Image.open("photo.jpg").convert("RGB"), dtype=np.float64) / 255.0
image = srgb_to_oklab(rgb)

# Extract a palette by k-means clustering in Oklab.
palette, shares = extract_palette(image, n_colours=8, seed=0)

# Build a global shaping matrix from the palette's own covariance.
M, eigenvalues, gains = global_shaping_matrix(palette)

# Dither with shaped error diffusion.
dithered, indices = diffuse(image, palette, shaping=M)

# Convert back to sRGB for saving or display.
out = np.clip(oklab_to_srgb(dithered), 0, 1)
Image.fromarray((out * 255).round().astype(np.uint8)).save("dithered.png")
```

## The shaping matrix

`global_shaping_matrix` computes one matrix from the whole palette’s
covariance. It is the right default — and, on the evidence to date, the
right choice full stop: a 124-image corpus run (see
`experiments/dithering_quality/`) found that a per-cell variant holds up
on noise reduction but loses on tone accuracy and the low-frequency worm
metric from K=16 onward. `per_cell_shaping_matrices`
remains exported for completeness and for future palettes whose local
structure genuinely varies, but it is not the recommended path.

Passing `shaping=None` to `diffuse` reproduces plain vector Floyd-Steinberg;
this is also what an isotropic palette collapses to on its own, since every
gain approaches 1 when the palette has real extent along every axis.

## Reading a palette’s geometry before dithering

`eigenspectrum_report` gives an eigenvalue-and-gain summary of a palette,
useful for judging in advance how much shaping will do:

```python
from tramage import eigenspectrum_report

report = eigenspectrum_report(palette)
print(report)
```

A palette with one eigenvalue much smaller than the others is exactly the
regime shaping was built for. See docs/for-designers.md for what that means
in practice, and docs/mathematical-appendix.md for the proof underneath it.

## Choosing epsilon

`eps` sets the variance scale below which an axis is treated as degenerate.
The default, `0.02 ** 2`, is a reasonable starting point in Oklab units. See
`experiments/README.md` for the planned sweep across values, and
docs/mathematical-appendix.md for how it relates to a palette’s own
eigenvalues.
