"""Tests for the Kolpatzik-Bouman filter and diffusion function."""

import numpy as np
import pytest

from tramage.colour import srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.kb_filter import kb_diffuse, kb_filters


def test_kb_filters_shape_and_finite():
    f = kb_filters()
    assert f.weights.shape == (3, 4)
    assert np.all(np.isfinite(f.weights))
    assert f.channel_names == ("Y", "rg", "yb")


def test_kb_filters_not_all_equal_to_floyd_steinberg():
    """The published KB values differ from Floyd-Steinberg's 7/3/5/1 over 16.

    If they were identical, the KB baseline would silently reduce to FS and the
    experiment would be meaningless.
    """
    f = kb_filters()
    fs = np.array([7 / 16, 3 / 16, 5 / 16, 1 / 16])
    # At least the luminance (Y) channel should differ from FS.
    assert not np.allclose(f.weights[0], fs)


def test_kb_diffuse_output_shape_matches_input():
    rng = np.random.default_rng(0)
    image = srgb_to_oklab(rng.uniform(size=(32, 32, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(8, 3)))
    dithered, indices = kb_diffuse(image, palette)
    assert dithered.shape == image.shape
    assert indices.shape == (32, 32)


def test_kb_diffuse_differs_from_plain_floyd_steinberg():
    """The KB filter taps are different from FS.

    So the output of kb_diffuse with the published filters must differ from
    diffuse with FS weights.
    """
    rng = np.random.default_rng(1)
    image = srgb_to_oklab(rng.uniform(size=(48, 48, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(6, 3)))
    fs_out, _ = diffuse(image, palette, shaping=None)
    kb_out, _ = kb_diffuse(image, palette)
    assert not np.allclose(fs_out, kb_out)


def test_kb_diffuse_constant_image_gives_constant_output():
    """A flat image made entirely of one palette color should dither losslessly.

    The input is exactly a palette color, so every pixel quantizes with zero
    error and no error ever enters the loop; this pins lossless reproduction,
    and no biased kernel could be caught by it.
    """
    colour = np.array([0.5, 0.0, 0.0])
    image = np.tile(colour, (16, 16, 1))
    palette = np.array([colour, colour + np.array([0.1, 0.0, 0.0])])
    dithered, _ = kb_diffuse(image, palette)
    assert np.allclose(dithered, image)


def test_kb_diffuse_with_shaping_differs_from_without():
    """The shaping matrix should change the output of kb_diffuse.

    This confirms the shaping path is wired through the opponent-space
    diffusion loop.
    """
    rng = np.random.default_rng(2)
    image = srgb_to_oklab(rng.uniform(size=(40, 40, 3)))
    palette = srgb_to_oklab(rng.uniform(size=(5, 3)))
    # A shaping matrix that is not identity: scale the first channel by 0.5.
    M = np.diag([0.5, 1.0, 1.0])
    f = kb_filters()
    unshaped, _ = kb_diffuse(image, palette, f)
    shaped, _ = kb_diffuse(image, palette, f, shaping=M)
    assert not np.allclose(unshaped, shaped)


def test_kb_filters_match_published_table1():
    """Pin the exact Kolpatzik & Bouman (1992) Table 1 ("1st order LP") coefficients.

    Weights are listed in the FS support order ((0,1), (1,-1), (1,0), (1,1)).
    The KB baseline is only meaningful if these are the paper's values; a
    silent edit must fail loudly.
    """
    expected = np.array([
        [0.7770, -0.009, 0.7861, -0.6098],   # luminance (Y)
        [0.8767, 0.0359, 0.8205, -0.7376],   # chrominance (rg)
        [0.8767, 0.0359, 0.8205, -0.7376],   # chrominance (yb)
    ])
    np.testing.assert_allclose(kb_filters().weights, expected, atol=1e-12)


def test_kb_diffuse_rejects_bad_image_shape():
    """A non-(H, W, 3) image raises a clear ValueError at the boundary."""
    palette = np.array([[0.5, 0.0, 0.0], [0.6, 0.0, 0.0]])
    with pytest.raises(ValueError, match="shape"):
        kb_diffuse(np.zeros((8, 8, 4)), palette)


def test_kb_diffuse_rejects_empty_palette():
    """An empty palette raises a clear ValueError at the boundary."""
    with pytest.raises(ValueError, match="n >= 1"):
        kb_diffuse(np.zeros((8, 8, 3)), np.zeros((0, 3)))


def test_kb_diffuse_rejects_nonfinite_palette():
    """A palette containing NaN raises a clear ValueError at the boundary."""
    palette = np.array([[0.5, 0.0, 0.0], [np.nan, 0.0, 0.0]])
    with pytest.raises(ValueError, match="non-finite"):
        kb_diffuse(np.zeros((8, 8, 3)), palette)
