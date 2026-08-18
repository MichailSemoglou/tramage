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
XYZ_TO_OPP: np.ndarray = np.array([
    [0.2787, 0.7218, -0.1066],
    [-0.4486, 0.2898, 0.0777],
    [0.0860, -0.5898, 0.5011],
])
OPP_TO_XYZ: np.ndarray = np.linalg.inv(XYZ_TO_OPP)

# S-CIELAB separable-filter sigmas in degrees of visual angle, from the
# published Gaussian fits (luminance ~0.0283 deg, chrominance ~0.116 deg).
SIGMA_LUM_DEG = 0.0283
SIGMA_CHROM_DEG = 0.116


def pixels_per_degree(dpi: float, viewing_distance_cm: float) -> float:
    """Number of pixels per degree of visual angle at the given geometry.

    One pixel subtends ``atan(2.54 / dpi / viewing_distance_cm)`` radians;
    the small-angle approximation is used since the angle is below 0.02 rad
    for any practical display.
    """
    return (dpi * viewing_distance_cm) / (2.54 * (180.0 / np.pi))
