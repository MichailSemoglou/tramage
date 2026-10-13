"""Tests for the Damera-Venkata matrix-valued filter and diffusion function."""

import numpy as np
import pytest

from tramage.colour import srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.dv_filter import (
    DVFilters,
    _oklab_to_opponent,
    _opponent_to_oklab,
    dv_diffuse,
    dv_filters,
)
from tramage.shaping import global_shaping_matrix


def test_dv_filters_returns_correct_shapes():
    rng = np.random.default_rng(0)
    palette = srgb_to_oklab(rng.uniform(size=(8, 3)))
    f = dv_filters(palette, seed=1)
    assert isinstance(f, DVFilters)
    assert f.weights.shape == (4, 3, 3)
    assert f.K.shape == (3, 3)
    assert np.all(np.isfinite(f.weights))
    assert np.all(np.isfinite(f.K))
    assert f.display_dpi == 100.0
    assert f.viewing_distance_cm == 40.0


def test_dv_filters_taps_are_not_identity():
    """The designed matrix taps should differ from the uniform DC-preserving split.

    The trivial geometry-blind split is identity/4 per tap; the design
    should exploit palette geometry instead.
    """
    rng = np.random.default_rng(1)
    palette = srgb_to_oklab(rng.uniform(size=(6, 3)))
    f = dv_filters(palette, seed=2)
    identity_quarter = np.eye(3) / 4
    assert not np.allclose(f.weights, identity_quarter)


def test_dv_filters_single_colour_raises():
    palette = srgb_to_oklab(np.array([[0.5, 0.5, 0.5]]))
    with pytest.raises(ValueError, match="at least 2"):
        dv_filters(palette)


def test_dv_diffuse_output_shape_matches_input():
    rng = np.random.default_rng(3)
    image = srgb_to_oklab(rng.uniform(size=(32, 32, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(8, 3)))
    f = dv_filters(palette, seed=4)
    dithered, indices = dv_diffuse(image, palette, f)
    assert dithered.shape == image.shape
    assert indices.shape == (32, 32)


def test_dv_diffuse_output_uses_only_palette_colours():
    rng = np.random.default_rng(5)
    image = srgb_to_oklab(rng.uniform(size=(16, 16, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(4, 3)))
    f = dv_filters(palette, seed=6)
    dithered, indices = dv_diffuse(image, palette, f)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))
    # dv_diffuse returns palette_oklab[indices], an exact gather, so this
    # tolerance is defensive rather than a bound on numerical error.
    assert np.allclose(dithered, palette[indices], atol=1e-4)


def test_dv_diffuse_exact_palette_colour_reproduces_it():
    """A flat image made entirely of one palette color should dither losslessly."""
    palette = srgb_to_oklab(np.array([[0.6, 0.1, 0.3], [0.2, 0.8, 0.5]]))
    image = np.tile(palette[0], (10, 10, 1))
    f = dv_filters(palette, seed=7)
    dithered, indices = dv_diffuse(image, palette, f)
    assert np.all(indices == 0)
    assert np.allclose(dithered, image, atol=1e-4)


def test_dv_diffuse_differs_from_plain_floyd_steinberg():
    """The DV matrix-valued taps differ from scalar FS weights, so output must differ."""
    rng = np.random.default_rng(8)
    image = srgb_to_oklab(rng.uniform(size=(48, 48, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(6, 3)))
    f = dv_filters(palette, seed=9)
    fs_out, _ = diffuse(image, palette, shaping=None)
    dv_out, _ = dv_diffuse(image, palette, f)
    assert not np.allclose(fs_out, dv_out, atol=1e-6)


def test_dv_diffuse_auto_designs_filter_when_none():
    """dv_diffuse with filters=None should design a filter on the fly."""
    rng = np.random.default_rng(10)
    image = srgb_to_oklab(rng.uniform(size=(32, 32, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(6, 3)))
    dithered, indices = dv_diffuse(image, palette, filters=None)
    assert dithered.shape == image.shape
    assert indices.shape == (32, 32)


def test_dv_diffuse_with_shaping():
    """A non-identity shaping matrix should change dv_diffuse output."""
    rng = np.random.default_rng(11)
    image = srgb_to_oklab(rng.uniform(size=(40, 40, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(5, 3)))
    f = dv_filters(palette, seed=12)
    M = np.diag([0.5, 1.0, 1.0])
    unshaped, _ = dv_diffuse(image, palette, f)
    shaped, _ = dv_diffuse(image, palette, f, shaping=M)
    assert not np.allclose(unshaped, shaped, atol=1e-6)


def test_dv_diffuse_rejects_per_cell_shaping():
    """dv_diffuse does not support (n, 3, 3) per-cell shaping matrices."""
    rng = np.random.default_rng(13)
    image = srgb_to_oklab(rng.uniform(size=(16, 16, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(4, 3)))
    f = dv_filters(palette, seed=14)
    per_cell = np.zeros((4, 3, 3))
    with pytest.raises(ValueError, match="only supports global shaping"):
        dv_diffuse(image, palette, f, shaping=per_cell)


def test_dv_diffuse_serpentine_false():
    """Raster-order DV diffusion should produce valid output."""
    rng = np.random.default_rng(15)
    image = srgb_to_oklab(rng.uniform(size=(16, 16, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(5, 3)))
    f = dv_filters(palette, seed=16)
    dithered, indices = dv_diffuse(image, palette, f, serpentine=False)
    assert dithered.shape == image.shape
    assert indices.shape == (16, 16)
    assert set(np.unique(indices)).issubset(set(range(len(palette))))


def test_opponent_round_trip():
    """Oklab -> opponent -> Oklab should be a round trip (within tolerance)."""
    rng = np.random.default_rng(17)
    lab = srgb_to_oklab(rng.uniform(size=(200, 3)))
    opp = _oklab_to_opponent(lab)
    back = _opponent_to_oklab(opp)
    assert np.allclose(lab, back, atol=1e-4)


def test_dv_filters_reproducible():
    """Same seed gives identical filters."""
    rng = np.random.default_rng(18)
    palette = srgb_to_oklab(rng.uniform(size=(8, 3)))
    f1 = dv_filters(palette, seed=42)
    f2 = dv_filters(palette, seed=42)
    assert np.allclose(f1.weights, f2.weights)
    assert np.allclose(f1.K, f2.K)


def test_dv_filters_different_palettes_different_filters():
    """Different palettes should produce different filter taps."""
    rng = np.random.default_rng(19)
    p1 = srgb_to_oklab(rng.uniform(size=(8, 3)))
    p2 = srgb_to_oklab(rng.uniform(size=(8, 3)) * 0.5 + 0.5)
    f1 = dv_filters(p1, seed=20)
    f2 = dv_filters(p2, seed=20)
    assert not np.allclose(f1.weights, f2.weights)


def test_dv_filters_custom_geometry():
    """Filters designed for different viewing geometry should differ."""
    rng = np.random.default_rng(21)
    palette = srgb_to_oklab(rng.uniform(size=(8, 3)))
    f1 = dv_filters(palette, display_dpi=72, viewing_distance_cm=30, seed=22)
    f2 = dv_filters(palette, display_dpi=200, viewing_distance_cm=80, seed=22)
    # Different geometries give different CSF sigmas -> different filters.
    assert not np.allclose(f1.weights, f2.weights)


def test_dv_diffuse_with_shaping_and_degenerate_palette():
    """Shaping is inert on an exactly rank-deficient palette, even in DV diffusion.

    Every palette color shares the same b, so the b part of the buffer state
    contributes an identical constant to every candidate's squared distance
    and can never change which color is nearest, and the shaping matrix
    annihilates the b error exactly (zero eigenvalue, zero gain). The image
    adds off-plane demand in b, so there is real error to shape: the
    unshaped run diffuses the full error through the matrix-valued taps, the
    shaped run a b-free one, and the two must still choose the same palette
    index at every pixel. Without the off-plane demand every pixel would
    quantize with zero error and the assertion would pass vacuously.
    """
    b_plane = 0.02
    Ls = [0.40, 0.55, 0.70]
    As = [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, b_plane] for L in Ls for a in As])

    rng = np.random.default_rng(23)
    image = palette[rng.integers(0, len(palette), size=(20, 20))].astype(np.float64)
    image[..., 2] += rng.normal(scale=0.05, size=(20, 20))  # off-plane demand

    f = dv_filters(palette, seed=24)
    M, _, _ = global_shaping_matrix(palette, eps=0.02 ** 2)
    unshaped, unshaped_idx = dv_diffuse(image, palette, f)
    shaped, shaped_idx = dv_diffuse(image, palette, f, shaping=M)

    assert np.array_equal(unshaped_idx, shaped_idx)
    assert np.array_equal(unshaped, shaped)
