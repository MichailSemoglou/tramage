"""Palette extraction and eigenspectrum reporting.

Two things a palette-aware method needs before it can shape anything: a
palette, and an honest account of that palette's geometry. This module
extracts palettes from images by k-means clustering in Oklab, and reports
the eigenspectrum that :mod:`tramage.shaping` will act on, so the degree of
degeneracy (and therefore how much shaping will do) is visible before any
dithering happens.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.cluster.vq import kmeans2

from .colour import FloatArray
from .shaping import DEFAULT_EPS, global_shaping_matrix, per_cell_shaping_matrices


def extract_palette(
    image: FloatArray,
    n_colours: int,
    seed: int | None = None,
    missing: str = "warn",
) -> tuple[FloatArray, FloatArray]:
    """Extract a palette from an Oklab image by k-means clustering.

    Parameters
    ----------
    image:
        Oklab image, shape ``(H, W, 3)``.
    n_colours:
        Number of palette colors to extract.
    seed:
        Seed for the k-means initialization, for reproducible palettes.
    missing:
        How to handle an empty cluster (`n_colours` larger than the image's
        actual color diversity produces one). ``"warn"`` (the default)
        keeps that centroid at its initial position with a 0% share and
        prints a warning; ``"raise"`` raises instead, if a duplicate,
        0%-share color in the palette would be a problem for the caller.

    Returns
    -------
    A tuple ``(palette, shares)``: the cluster centroids in Oklab, ordered
    by descending share of pixels, and each color's share as a fraction of
    the total pixel count.

    """
    pixels = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    centroids, labels = kmeans2(pixels, n_colours, minit="++", seed=seed, missing=missing)
    shares = np.bincount(labels, minlength=n_colours) / len(labels)
    order = np.argsort(-shares)
    return centroids[order], shares[order]


@dataclass
class EigenspectrumReport:
    """Palette geometry as seen by the shaping step.

    Attributes
    ----------
    eigenvalues:
        Eigenvalues of the whole palette's covariance in Oklab, ascending.
    gains:
        Global shaping gain for each eigenvalue, ``lambda / (lambda + eps)``.
    per_cell_gain_min, per_cell_gain_max:
        Elementwise minimum and maximum, across palette colors, of the
        per-cell gains from :func:`tramage.shaping.per_cell_shaping_matrices`.
        A wide range between them means the palette's local structure
        varies across the gamut; a narrow range means the palette is
        roughly self-similar everywhere. When :func:`eigenspectrum_report`
        is called with ``per_cell=False``, both are filled with zeros: a
        documented fallback for the skipped computation, not measured
        gains.
    eps:
        Variance scale below which an axis is treated as degenerate; the
        value the gains were computed with.

    """

    eigenvalues: FloatArray
    gains: FloatArray
    per_cell_gain_min: FloatArray
    per_cell_gain_max: FloatArray
    eps: float = DEFAULT_EPS

    def __str__(self) -> str:
        lines = [
            f"global eigenvalues: {np.round(self.eigenvalues, 6)}",
            f"global gains:       {np.round(self.gains, 3)}",
            f"per-cell gain range: min {np.round(self.per_cell_gain_min, 3)}"
            f"  max {np.round(self.per_cell_gain_max, 3)}",
        ]
        return "\n".join(lines)


def eigenspectrum_report(
    palette: FloatArray,
    k: int = 4,
    eps: float = DEFAULT_EPS,
    *,
    per_cell: bool = True,
) -> EigenspectrumReport:
    """Build an :class:`EigenspectrumReport` for a palette.

    Reports both the global covariance eigenspectrum and the range of
    per-cell gains across the palette, so the same near-degeneracy that
    :mod:`tramage.shaping` will act on is visible ahead of time.

    Parameters
    ----------
    palette:
        Palette colors in Oklab, shape ``(n, 3)``.
    k:
        Number of neighbors for per-cell shaping matrices.
    eps:
        Variance scale below which an axis is treated as degenerate.
    per_cell:
        If ``True`` (default), also computes per-cell shaping matrices so
        ``per_cell_gain_min`` and ``per_cell_gain_max`` are populated.
        Set ``False`` to skip the O(n²) per-cell cost when only the
        global eigenspectrum is needed; the two per-cell fields are then
        zero-filled, a documented fallback rather than measured data.

    """
    _, eigenvalues, gains = global_shaping_matrix(palette, eps=eps)
    if per_cell:
        _, _, cell_gains = per_cell_shaping_matrices(palette, k=k, eps=eps)
        cell_gains_arr = np.array(cell_gains)
    else:
        cell_gains_arr = np.zeros((len(np.asarray(palette)), 3))
    return EigenspectrumReport(
        eigenvalues=eigenvalues,
        gains=gains,
        per_cell_gain_min=cell_gains_arr.min(axis=0),
        per_cell_gain_max=cell_gains_arr.max(axis=0),
        eps=eps,
    )


# Gains at or below this are treated as "flat" for describe_palette, and
# gains at or above this are treated as "real extent". Between the two is
# reported as "partial". These are descriptive bands for plain-language
# reporting, not a tuning recommendation for eps itself. The values 0.3
# and 0.7 are chosen so that directions whose variance is within roughly
# a factor of two of eps (gain ≈ 0.5 at variance = eps) are described as
# "partial" rather than given a definitive label, while directions clearly
# above or below that threshold get a clear characterization.
_FLAT_GAIN = 0.3
_REAL_EXTENT_GAIN = 0.7


def describe_palette(
    palette: FloatArray,
    k: int = 4,
    eps: float = DEFAULT_EPS,
) -> str:
    """Plain-language account of a palette's geometry for shaping.

    Wraps :func:`eigenspectrum_report` in a short, honestly worded
    paragraph: how many of the palette's three directions in Oklab have
    real extent versus are close to flat, what that implies for how much
    work shaping is likely to do, and whether the palette's local structure
    is uniform enough that the simpler global variant should behave like
    the per-cell one.

    This describes the palette's geometry relative to the `eps` supplied
    (the default if none is given). It does not recommend a value for
    `eps`; choosing one well is an open question this project is still
    characterizing (see docs/mathematical-appendix.md and
    experiments/README.md), and asserting a tuning rule here would get
    ahead of that work.
    """
    report = eigenspectrum_report(palette, k=k, eps=eps)
    n = len(np.asarray(palette))

    n_flat = int(np.sum(report.gains <= _FLAT_GAIN))
    n_real = int(np.sum(report.gains >= _REAL_EXTENT_GAIN))

    if n_flat == 0:
        shape_sentence = (
            f"This {n}-color palette has real extent in all three directions "
            f"in Oklab, relative to eps={eps:g}."
        )
        effect_sentence = (
            "Shaping is expected to have little to do here: the shaping matrix "
            "is close to the identity, so output should be close to ordinary "
            "vector Floyd-Steinberg."
        )
    elif n_real == 0:
        shape_sentence = (
            f"This {n}-color palette has almost no measurable spread in any "
            f"direction in Oklab, relative to eps={eps:g}."
        )
        effect_sentence = (
            "There is little error for either shaped or unshaped diffusion to "
            "redistribute in the first place, so the two should behave "
            "similarly."
        )
    else:
        shape_sentence = (
            f"This {n}-color palette has real extent in {n_real} of 3 "
            f"directions in Oklab, with {n_flat} close to flat, relative to "
            f"eps={eps:g}."
        )
        effect_sentence = (
            "Shaping is likely to matter here: error the palette cannot "
            "represent along the flat direction will be damped rather than "
            "left to drift, at some cost to fidelity in whatever the image "
            "asks of that direction. Whether that trade is worthwhile "
            "depends on the image, not only the palette."
        )

    cell_spread = float(np.max(report.per_cell_gain_max - report.per_cell_gain_min))
    if cell_spread > _FLAT_GAIN:
        locality_sentence = (
            "The palette's local structure varies noticeably across its "
            "gamut, so per-cell shaping may behave differently from the "
            "global variant in different regions."
        )
    else:
        locality_sentence = (
            "Local structure is fairly uniform across the palette, so the "
            "simpler global variant should behave similarly to per-cell "
            "shaping."
        )

    return " ".join([shape_sentence, effect_sentence, locality_sentence])
