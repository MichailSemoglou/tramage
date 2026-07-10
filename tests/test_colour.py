"""Round-trip tests for sRGB <-> Oklab conversion."""

import warnings

import numpy as np

from tramage.colour import (
    cielab_to_srgb,
    in_gamut,
    oklab_to_srgb,
    srgb_to_cielab,
    srgb_to_linear,
    srgb_to_oklab,
)


def test_round_trip_grid_of_colours():
    rng = np.random.default_rng(0)
    rgb = rng.uniform(0.0, 1.0, size=(200, 3))
    lab = srgb_to_oklab(rgb)
    back = oklab_to_srgb(lab)
    assert np.allclose(rgb, back, atol=1e-8)


def test_round_trip_preserves_shape():
    rng = np.random.default_rng(1)
    rgb = rng.uniform(0.0, 1.0, size=(8, 5, 3))
    lab = srgb_to_oklab(rgb)
    assert lab.shape == rgb.shape
    back = oklab_to_srgb(lab)
    assert back.shape == rgb.shape


def test_black_and_white_round_trip():
    rgb = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lab = srgb_to_oklab(rgb)
    back = oklab_to_srgb(lab)
    assert np.allclose(rgb, back, atol=1e-8)


def test_in_gamut_true_for_srgb_primaries():
    rgb = np.eye(3)
    lab = srgb_to_oklab(rgb)
    assert np.all(in_gamut(lab))


def test_in_gamut_false_outside_srgb_cube():
    # Push a valid colour's lightness far out of range in Oklab.
    lab = srgb_to_oklab(np.array([0.5, 0.5, 0.5]))
    out_of_gamut = lab.copy()
    out_of_gamut[0] = 5.0
    assert not in_gamut(out_of_gamut)


def test_srgb_to_linear_no_warning_for_negative_input():
    """Negative input beyond -0.055 previously raised a spurious RuntimeWarning
    from the unused branch of np.where, even though the selected result was
    already correct.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = srgb_to_linear(np.array([-0.5, 0.5, 1.2]))
    assert np.isclose(result[0], -0.5 / 12.92)


def test_in_gamut_scalar_vs_batch():
    rgb = np.array([[0.2, 0.4, 0.6], [0.9, 0.1, 0.3]])
    lab = srgb_to_oklab(rgb)
    batch_result = in_gamut(lab)
    scalar_results = np.array([in_gamut(lab[i]) for i in range(len(lab))])
    assert batch_result.shape == (2,)
    assert np.array_equal(batch_result, scalar_results)


# --- CIELAB (D65) --------------------------------------------------------- #
# These mirror the Oklab tests above. CIELAB is added to support the
# Oklab-vs-CIELAB head-to-head in experiments/colour_space/, not as the
# package's recommended space; the package default remains Oklab.

def test_cielab_round_trip_grid_of_colours():
    rng = np.random.default_rng(20)
    rgb = rng.uniform(0.0, 1.0, size=(200, 3))
    lab = srgb_to_cielab(rgb)
    back = cielab_to_srgb(lab)
    assert np.allclose(rgb, back, atol=1e-8)


def test_cielab_round_trip_preserves_shape():
    rng = np.random.default_rng(21)
    rgb = rng.uniform(0.0, 1.0, size=(8, 5, 3))
    lab = srgb_to_cielab(rgb)
    assert lab.shape == rgb.shape
    back = cielab_to_srgb(lab)
    assert back.shape == rgb.shape


def test_cielab_black_and_white_round_trip():
    rgb = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    lab = srgb_to_cielab(rgb)
    back = cielab_to_srgb(lab)
    assert np.allclose(rgb, back, atol=1e-8)


def test_cielab_midgray_anchor():
    """sRGB [0.5, 0.5, 0.5] maps to L* ~ 53.39, the standard published value
    for mid-gray under D65. Anchors the implementation to an external
    reference, not just round-trip closure (which a sign-flipped matrix
    could also pass)."""
    lab = srgb_to_cielab(np.array([0.5, 0.5, 0.5]))
    assert np.isclose(lab[0], 53.39, atol=0.05)
    assert np.isclose(lab[1], 0.0, atol=0.01)
    assert np.isclose(lab[2], 0.0, atol=0.01)
