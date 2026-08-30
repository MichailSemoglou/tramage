"""Shared physical and perceptual-model constants for tramage.

Single source of truth for the colour-science values that more than one
module needs: the sRGB<->XYZ matrices, the D65 white point, the CIELAB
nonlinearity constants, the Poirson-Wandell opponent transform, and the
S-CIELAB spatial-filter sigmas. Keeping them here prevents the copies in
``colour``, ``metrics``, and ``dv_filter`` from drifting apart.
"""

from __future__ import annotations

import numpy as np

# sRGB (linear) -> XYZ for a D65 white point (IEC 61966-2-1).
SRGB_TO_XYZ_D65: np.ndarray = np.array([
    [0.4123908, 0.3575843, 0.1804808],
    [0.2126390, 0.7151687, 0.0721923],
    [0.0193308, 0.1191948, 0.9505322],
])
XYZ_TO_LINEAR_SRGB: np.ndarray = np.linalg.inv(SRGB_TO_XYZ_D65)

# D65 reference white in absolute XYZ (2-degree observer).
D65_XN = 0.95047
D65_YN = 1.00000
D65_ZN = 1.08883

# CIELAB cube-root threshold and linear-branch coefficient.
EPS_LAB = (6.0 / 29.0) ** 3
KAPPA_LAB = (1.0 / 3.0) * (29.0 / 6.0) ** 2

# Poirson & Wandell (1996) pattern-colour separable opponent matrix, XYZ D65
# -> opponent (O1 luminance, O2 red-green, O3 blue-yellow).
# Note: Zhang & Wandell (1997, p. 62) print the O2 row with -0.077 for the Z
# coefficient, but that sign is inconsistent with the paper's own printed
# inverse; the canonical SCIELAB-1996/ISET implementation uses +0.077, as
# here. Worth a footnote in the paper to pre-empt the reviewer check.
XYZ_TO_OPP: np.ndarray = np.array([
    [0.2787, 0.7218, -0.1066],
    [-0.4486, 0.2898, 0.0777],
    [0.0860, -0.5898, 0.5011],
])
OPP_TO_XYZ: np.ndarray = np.linalg.inv(XYZ_TO_OPP)

# S-CIELAB spatial-filter parameters (Zhang & Wandell 1997, Appendix A).
# Each opponent channel's pre-filter is a weighted sum of unit-volume 2D
# Gaussians with these spreads (degrees of visual angle), the composite
# normalised to unit DC gain. The luminance filter's negative wide lobe
# gives the bandpass surround of the luminance CSF.
SCIELAB_LUM_WEIGHTS = (0.921, 0.105, -0.108)
SCIELAB_LUM_SPREADS_DEG = (0.0283, 0.133, 4.336)
SCIELAB_RG_WEIGHTS = (0.531, 0.330)
SCIELAB_RG_SPREADS_DEG = (0.0392, 0.494)
SCIELAB_BY_WEIGHTS = (0.488, 0.371)
SCIELAB_BY_SPREADS_DEG = (0.0536, 0.386)

# Legacy single-Gaussian approximation of the S-CIELAB filters: luminance
# keeps only the narrowest published lobe; chrominance collapses both
# channels to one shared spread. Retained for the ``filters="gaussian"``
# comparison path in ``metrics.scielab_de`` and for the (disabled) DV
# filter design in ``dv_filter``.
SIGMA_LUM_DEG = 0.0283
SIGMA_CHROM_DEG = 0.116


def pixels_per_degree(dpi: float, viewing_distance_cm: float) -> float:
    """Number of pixels per degree of visual angle at the given geometry.

    One pixel subtends ``atan(2.54 / dpi / viewing_distance_cm)`` radians;
    the small-angle approximation is used since the angle is below 0.02 rad
    for any practical display.
    """
    return (dpi * viewing_distance_cm) / (2.54 * (180.0 / np.pi))
