"""Tests for shaping-matrix construction: the isotropic limit and inertness."""

import numpy as np
import pytest

from tramage.shaping import (
    gain,
    global_shaping_matrix,
    per_cell_shaping_matrices,
    shaping_matrix,
)


def test_isotropic_palette_gives_identity_matrix():
    """An isotropic palette should push all gains to ~1, M to ~identity.

    This is the limiting case in which palette-structure-aware shaping
    collapses exactly to vector Floyd-Steinberg: it contains the baseline.
    """
    rng = np.random.default_rng(2)
    # A dense, isotropic cloud of points: covariance is close to a multiple
    # of the identity, and its scale is large relative to eps.
    palette = rng.normal(loc=[0.5, 0.0, 0.0], scale=0.2, size=(500, 3))
    eps = 1e-8
    M, eigenvalues, gains = global_shaping_matrix(palette, eps=eps)
    assert np.allclose(gains, 1.0, atol=1e-3)
    assert np.allclose(M, np.eye(3), atol=1e-2)


def test_inertness_exactly_degenerate_palette():
    """A rank-deficient palette's degenerate axis gets exactly zero gain.

    For a palette lying exactly on a plane in Oklab, the eigenvalue along
    the plane's normal is exactly zero, so its gain lambda / (lambda + eps)
    is exactly zero for any eps > 0. Shaping therefore projects out that
    axis completely: error orthogonal to the palette's affine hull is
    annihilated rather than merely damped.
    """
    b_plane = 0.02
    Ls = [0.40, 0.55, 0.70]
    As = [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, b_plane] for L in Ls for a in As])

    M, eigenvalues, gains = global_shaping_matrix(palette, eps=0.02 ** 2)

    assert np.isclose(eigenvalues.min(), 0.0, atol=1e-12)
    assert np.isclose(gains.min(), 0.0, atol=1e-12)

    # The degenerate direction here is the b axis, since every colour sits
    # on b = b_plane. Error purely along b should be mapped to (near) zero.
    off_plane_error = np.array([0.0, 0.0, 1.0])
    shaped = M @ off_plane_error
    assert np.allclose(shaped, 0.0, atol=1e-10)


def test_per_cell_matches_global_for_isotropic_palette():
    """Per-cell shaping should also reach ~identity for a locally isotropic palette."""
    rng = np.random.default_rng(3)
    palette = rng.normal(loc=[0.5, 0.0, 0.0], scale=0.2, size=(60, 3))
    matrices, eigenvalues, gains = per_cell_shaping_matrices(palette, k=8, eps=1e-8)
    assert matrices.shape == (60, 3, 3)
    for M in matrices:
        assert np.allclose(M, np.eye(3), atol=1e-1)


def test_shaping_matrix_is_symmetric():
    rng = np.random.default_rng(4)
    points = rng.normal(size=(30, 3))
    M, _, _ = shaping_matrix(points)
    assert np.allclose(M, M.T)


def test_per_cell_shaping_rejects_single_colour_palette():
    """A palette of 1 colour has no neighbours to build a local matrix from.

    Previously this silently produced a NaN covariance and failed deep
    inside `np.linalg.eigh` with a confusing `LinAlgError`; it should raise
    a clear, immediate error instead.
    """
    palette = np.array([[0.5, 0.0, 0.0]])
    with pytest.raises(ValueError, match="at least 2"):
        per_cell_shaping_matrices(palette, k=4)


def test_per_cell_shaping_rejects_k_of_zero():
    """k=0 would otherwise resolve to an empty neighbour set; reject it explicitly."""
    rng = np.random.default_rng(13)
    palette = rng.normal(size=(10, 3))
    with pytest.raises(ValueError, match="k must be"):
        per_cell_shaping_matrices(palette, k=0)


def test_gain_rejects_non_positive_eps():
    """eps=0 with an exactly-zero eigenvalue previously produced a silent NaN gain."""
    with pytest.raises(ValueError):
        gain(np.array([0.0, 1e-4, 0.5]), eps=0.0)
    with pytest.raises(ValueError):
        gain(np.array([0.0, 1e-4, 0.5]), eps=-1.0)


def test_shaping_matrix_rejects_empty_points():
    """Calling shaping_matrix on an empty array should raise a clear error,
    not a cryptic division-by-zero."""
    with pytest.raises(ValueError, match="at least one point"):
        shaping_matrix(np.empty((0, 3)))
