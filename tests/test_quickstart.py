"""The documented quickstart example runs end to end.

Mirrors docs/quickstart.md's minimal example with a synthetic image in place
of the photo file, so a rename of any public import it uses fails here first.
"""

import numpy as np

from tramage import (
    diffuse,
    extract_palette,
    global_shaping_matrix,
    oklab_to_srgb,
    srgb_to_oklab,
)


def test_quickstart_minimal_example_runs():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.0, size=(32, 32, 3))

    image = srgb_to_oklab(rgb)
    palette, shares = extract_palette(image, n_colours=8, seed=0)
    M, eigenvalues, gains = global_shaping_matrix(palette)
    dithered, indices = diffuse(image, palette, shaping=M)
    out = np.clip(oklab_to_srgb(dithered), 0, 1)

    assert palette.shape == (8, 3)
    assert shares.shape == (8,)
    assert M.shape == (3, 3)
    assert eigenvalues.shape == (3,)
    assert gains.shape == (3,)
    assert dithered.shape == image.shape
    assert indices.shape == (32, 32)
    assert out.shape == rgb.shape
    assert np.isfinite(out).all()
