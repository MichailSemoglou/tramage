"""Tests for serpentine error diffusion: weight energy and basic behaviour."""

import numpy as np

from tramage.diffusion import DEFAULT_WEIGHTS, _scanlines, diffuse
from tramage.shaping import global_shaping_matrix, per_cell_shaping_matrices


def test_default_weights_conserve_energy():
    """The Floyd-Steinberg kernel must distribute exactly 100% of the error.

    If the weights summed to anything other than 1, error diffusion would
    either manufacture or destroy energy at every pixel, and systematic
    bias would accumulate even for a palette that can represent the image
    exactly.
    """
    total = sum(weight for _, _, weight in DEFAULT_WEIGHTS)
    assert np.isclose(total, 1.0)


def test_diffuse_output_uses_only_palette_colours():
    rng = np.random.default_rng(5)
    image = rng.uniform(size=(16, 16, 3))
    palette = rng.uniform(size=(4, 3))
    dithered, indices = diffuse(image, palette)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))
    assert np.allclose(dithered, palette[indices])


def test_diffuse_exact_palette_colour_reproduces_it_perfectly():
    """A flat image made entirely of one palette colour should dither losslessly."""
    palette = np.array([[0.4, 0.0, 0.0], [0.6, 0.1, -0.1]])
    image = np.tile(palette[0], (10, 10, 1))
    dithered, indices = diffuse(image, palette)
    assert np.all(indices == 0)
    assert np.allclose(dithered, image)


def test_shaping_none_equals_identity_shaping():
    rng = np.random.default_rng(6)
    image = rng.uniform(size=(12, 12, 3))
    palette = rng.uniform(size=(5, 3))
    unshaped, indices_unshaped = diffuse(image, palette, shaping=None)
    identity_shaped, indices_identity = diffuse(image, palette, shaping=np.eye(3))
    assert np.array_equal(indices_unshaped, indices_identity)
    assert np.allclose(unshaped, identity_shaped)


def test_shaping_is_inert_at_the_diffusion_level_for_a_degenerate_palette():
    """Shaping cannot change quantisation decisions on an exactly rank-deficient palette.

    Every palette colour here shares the same b, so b-error contributes an
    identical constant to every candidate's squared distance and can never
    change which one is nearest. Baseline and shaped diffusion must
    therefore choose exactly the same palette index at every pixel, even
    though the shaped run propagates a different (b-free) error internally.
    This is the inertness lemma showing up end-to-end, not just in the
    shaping matrix itself.
    """
    b_plane = 0.02
    Ls = [0.40, 0.55, 0.70]
    As = [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, b_plane] for L in Ls for a in As])

    rng = np.random.default_rng(7)
    image = palette[rng.integers(0, len(palette), size=(20, 20))].astype(np.float64)
    image[..., 2] += rng.normal(scale=0.05, size=(20, 20))  # off-plane demand

    M, _, _ = global_shaping_matrix(palette, eps=0.02 ** 2)
    baseline, baseline_indices = diffuse(image, palette, shaping=None)
    shaped, shaped_indices = diffuse(image, palette, shaping=M)

    assert np.array_equal(baseline_indices, shaped_indices)
    assert np.array_equal(baseline, shaped)


def test_diffuse_with_per_cell_shaping():
    """Per-cell shaping matrices should produce valid palette-colour output."""
    rng = np.random.default_rng(15)
    image = rng.uniform(size=(16, 16, 3))
    palette = rng.uniform(size=(6, 3))
    matrices, _, _ = per_cell_shaping_matrices(palette, k=3)
    dithered, indices = diffuse(image, palette, shaping=matrices)
    assert dithered.shape == image.shape
    assert indices.shape == (16, 16)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))
    assert np.allclose(dithered, palette[indices])


def test_diffuse_serpentine_false():
    """Plain raster (non-serpentine) diffusion should produce valid output."""
    rng = np.random.default_rng(16)
    image = rng.uniform(size=(12, 12, 3))
    palette = rng.uniform(size=(4, 3))
    dithered, indices = diffuse(image, palette, serpentine=False)
    assert dithered.shape == image.shape
    assert indices.shape == (12, 12)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))


def test_diffuse_custom_weights():
    """Non-Floyd-Steinberg weights should produce valid output."""
    rng = np.random.default_rng(17)
    image = rng.uniform(size=(12, 12, 3))
    palette = rng.uniform(size=(4, 3))
    custom = ((0, 1, 0.5), (0, 2, 0.25), (1, -1, 0.15), (1, 0, 0.10))
    dithered, indices = diffuse(image, palette, weights=custom)
    assert dithered.shape == image.shape
    assert indices.shape == (12, 12)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))


def test_scanlines_serpentine():
    """Verify the serpentine scan direction alternates each row."""
    rows = list(_scanlines(4, 5, serpentine=True))
    assert rows[0][2] == 1   # row 0: left-to-right, sign +1
    assert rows[1][2] == -1  # row 1: right-to-left, sign -1
    assert rows[2][2] == 1   # row 2: left-to-right, sign +1
    assert rows[3][2] == -1  # row 3: right-to-left, sign -1


def test_scanlines_raster():
    """Non-serpentine scan should go left-to-right on every row."""
    rows = list(_scanlines(3, 4, serpentine=False))
    for y, xs, sign in rows:
        assert sign == 1
        assert list(xs) == list(range(4))
