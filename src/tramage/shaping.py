"""Construction of palette-structure-aware error-shaping matrices.

The central idea: a palette's own covariance in Oklab tells you which
directions in colour space it can actually represent. Where the palette has
real extent, quantisation error in that direction is payable, later pixels
can use it to correct the running total. Where the palette is flat or
nearly flat, error in that direction is not payable: it does not change
which palette colour is nearest, so propagating it only adds noise.

Given a set of palette points, compute the covariance matrix, eigendecompose
it, and shrink each eigenvalue to a gain via ``g(lambda) = lambda / (lambda +
eps)``. Reassembling ``V diag(g) V^T`` gives a symmetric matrix `M` that
passes error through along axes of real extent (``g -> 1``) and damps it
along near-degenerate axes (``g -> 0``). See docs/mathematical-appendix.md
for the inertness lemma this rests on, and the isotropic limiting case in
which `M` becomes the identity and the method collapses to vector
Floyd-Steinberg.

Two variants are provided:

- :func:`global_shaping_matrix`, a single matrix from the whole palette's
  covariance.
- :func:`per_cell_shaping_matrices`, one matrix per palette colour, built
  from that colour's `k` nearest palette neighbours. This adapts to
  palettes whose local structure varies across the gamut (a palette that is
  globally three-dimensional but locally flat in one region, for example).
"""

from __future__ import annotations

import warnings

import numpy as np
from numpy.typing import NDArray

from .colour import FloatArray

DEFAULT_EPS = 0.02 ** 2


def gain(eigenvalues: FloatArray, eps: float) -> FloatArray:
    """Shrinkage gain ``lambda / (lambda + eps)`` for each eigenvalue."""
    if eps <= 0:
        raise ValueError(f"eps must be > 0, got {eps!r}")
    return eigenvalues / (eigenvalues + eps)


def shaping_matrix(
    points: FloatArray,
    centre: FloatArray | None = None,
    eps: float = DEFAULT_EPS,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Covariance-eigenstructure shaping matrix for a set of points.

    Parameters
    ----------
    points:
        Colours in Oklab, shape ``(n, 3)``, whose covariance defines the
        shaping. For the global variant these are the whole palette; for
        the per-cell variant they are one colour's nearest neighbours.
    centre:
        Point to measure spread around. Defaults to the mean of `points`.
        The per-cell variant passes the palette colour itself, so the
        matrix reflects local extent around that colour rather than the
        mean of its neighbourhood.
    eps:
        Variance scale below which an axis is treated as degenerate.

    Returns
    -------
    A tuple ``(M, eigenvalues, gains)``: the shaping matrix, the
    eigenvalues of the covariance, and the gain applied to each.
    """
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    if n == 0:
        raise ValueError("shaping_matrix requires at least one point")
    d = points - (points.mean(axis=0) if centre is None else centre)
    cov = d.T @ d / n
    eigenvalues, V = np.linalg.eigh(cov)
    g = gain(eigenvalues, eps)
    M = V @ np.diag(g) @ V.T
    return M, eigenvalues, g


def global_shaping_matrix(
    palette: FloatArray,
    eps: float = DEFAULT_EPS,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Single shaping matrix from the whole palette's covariance in Oklab."""
    return shaping_matrix(palette, centre=None, eps=eps)


def per_cell_shaping_matrices(
    palette: FloatArray,
    k: int = 4,
    eps: float = DEFAULT_EPS,
) -> tuple[NDArray[np.float64], list[FloatArray], list[FloatArray]]:
    """One shaping matrix per palette colour, from its `k` nearest neighbours.

    Nearest neighbours in the palette itself stand in for a Delaunay
    neighbourhood, which is what local shaping would ideally use. Delaunay
    triangulation degenerates for rank-deficient palettes (all points on a
    plane or line), while k-nearest-neighbours does not, so it is the more
    robust choice here.

    Returns
    -------
    A tuple ``(matrices, eigenvalues, gains)``: `matrices` has shape
    ``(n, 3, 3)``; `eigenvalues` and `gains` are lists of per-colour arrays,
    each of length 3, in palette order.
    """
    palette = np.asarray(palette, dtype=np.float64)
    n = len(palette)
    if n < 2:
        raise ValueError("per-cell shaping requires at least 2 palette colours")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k!r}")
    k = min(k, n - 1)
    if k < 3:
        warnings.warn(
            f"per-cell shaping with only {k} neighbour(s) per colour "
            f"(palette has {n} colours) will produce low-rank local "
            f"covariances; per-cell shaping works best with at least "
            f"k+1 palette colours",
            stacklevel=2,
        )
    matrices = np.zeros((n, 3, 3))
    eigenvalues: list[FloatArray] = []
    gains: list[FloatArray] = []
    for i in range(n):
        d = np.linalg.norm(palette - palette[i], axis=1)
        n_needed = min(k + 1, n)
        neighbours = np.argpartition(d, n_needed - 1)[:n_needed]
        neighbours = neighbours[neighbours != i][:k]
        M, lam, g = shaping_matrix(palette[neighbours], centre=palette[i], eps=eps)
        matrices[i] = M
        eigenvalues.append(lam)
        gains.append(g)
    return matrices, eigenvalues, gains
