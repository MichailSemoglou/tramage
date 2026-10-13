"""Conversion between sRGB and the perceptual spaces Oklab and CIELAB.

Plus the CIE XYZ path that S-CIELAB and the DV filter design build on.

Oklab is a perceptually uniform color space designed so that Euclidean
distance approximates perceived color difference reasonably well across
the whole gamut. That property is what lets the rest of this package treat
palette geometry (covariance, eigenvalues, nearest neighbors) as a
meaningful description of how much visual room a palette actually spans,
rather than an artifact of whichever encoding the colors happened to
arrive in.

All functions operate on the last axis of an array, so a single color of
shape ``(3,)`` and an image of shape ``(H, W, 3)`` both work.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from ._constants import (
    D65_XN as _XN,
)
from ._constants import (
    D65_YN as _YN,
)
from ._constants import (
    D65_ZN as _ZN,
)
from ._constants import (
    EPS_LAB as _EPS_LAB,
)
from ._constants import (
    KAPPA_LAB as _KAPPA_LAB,
)
from ._constants import (
    SRGB_TO_XYZ_D65 as _SRGB_TO_XYZ_D65,
)
from ._constants import (
    XYZ_TO_LINEAR_SRGB as _XYZ_FROM_SRGB_INV,
)

FloatArray = NDArray[np.float64]

# sRGB (linear) -> LMS, from the Oklab reference implementation.
_SRGB_TO_LMS = np.array([
    [0.4122214708, 0.5363325363, 0.0514459929],
    [0.2119034982, 0.6806995451, 0.1073969566],
    [0.0883024619, 0.2817188376, 0.6299787005],
])

# LMS' (cube-rooted) -> Oklab.
_LMS_TO_OKLAB = np.array([
    [0.2104542553, 0.7936177850, -0.0040720468],
    [1.9779984951, -2.4285922050, 0.4505937099],
    [0.0259040371, 0.7827717662, -0.8086757660],
])

_LMS_FROM_SRGB_INV = np.linalg.inv(_SRGB_TO_LMS)
_OKLAB_TO_LMS = np.linalg.inv(_LMS_TO_OKLAB)


def srgb_to_linear(c: FloatArray) -> FloatArray:
    """Undo the sRGB transfer function, returning linear-light values."""
    c = np.asarray(c, dtype=np.float64)
    # np.where evaluates both branches for every element, so clip only the
    # copy fed to the power branch to avoid a fractional power of a negative
    # number for elements that will be selected from the other branch anyway.
    c_pos = np.clip(c, 0.0, None)
    return np.where(c <= 0.04045, c / 12.92, ((c_pos + 0.055) / 1.055) ** 2.4)


def linear_to_srgb(c: FloatArray) -> FloatArray:
    """Apply the sRGB transfer function to linear-light values.

    Negative inputs are clipped to zero before the transfer function,
    since they are not physically meaningful for display emission.
    """
    c = np.asarray(c, dtype=np.float64)
    c = np.clip(c, 0.0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


def srgb_to_oklab(rgb: FloatArray) -> FloatArray:
    """Convert gamma-encoded sRGB in [0, 1] to Oklab."""
    lin = srgb_to_linear(rgb)
    lms = lin @ _SRGB_TO_LMS.T
    lms_ = np.cbrt(lms)
    return lms_ @ _LMS_TO_OKLAB.T


def oklab_to_srgb(lab: FloatArray) -> FloatArray:
    """Convert Oklab back to gamma-encoded sRGB.

    Only the lower bound is clipped: :func:`linear_to_srgb` clips negative
    linear-light values to zero, while values above 1 pass through. Colors
    outside the sRGB gamut can therefore come back as out-of-range floats
    on the high side, so callers can decide how to handle them. For
    out-of-gamut detection use :func:`in_gamut`, which checks the
    linear-light values before the transfer function and so sees both
    bounds.
    """
    lms_ = lab @ _OKLAB_TO_LMS.T
    lms = lms_ ** 3
    lin = lms @ _LMS_FROM_SRGB_INV.T
    return linear_to_srgb(lin)


def in_gamut(lab: FloatArray, tol: float = 1e-6) -> np.bool_ | NDArray[np.bool_]:
    """Whether Oklab color(s) fall inside the sRGB gamut, within `tol`.

    Checks the linear-light RGB values prior to the sRGB transfer function,
    since that is where the gamut boundary is a simple unit cube.

    When `lab` is a 1-D array (a single color of shape ``(3,)``), returns
    a scalar ``np.bool_``. For any shape with ``ndim >= 2``, returns an
    array of per-color booleans of shape ``lab.shape[:-1]``: one boolean
    per color, so an image of shape ``(H, W, 3)`` yields an ``(H, W)`` map.
    """
    lab = np.asarray(lab, dtype=np.float64)
    lms_ = lab @ _OKLAB_TO_LMS.T
    lms = lms_ ** 3
    lin = lms @ _LMS_FROM_SRGB_INV.T
    inside = (lin >= -tol) & (lin <= 1.0 + tol)
    return np.all(inside) if lab.ndim == 1 else np.all(inside, axis=-1)


# --- CIELAB (D65) -----------------------------------------------------------
#
# CIELAB is the field's default perceptual space (Kolpatzik & Bouman 1995,
# Orchard & Bouman 1991, Damera-Venkata & Evans 2001 all use it). This
# conversion path exists so the head-to-head in experiments/colour_space/
# can test whether Oklab, the package default, is actually the right
# choice for palette-structure-aware shaping, rather than asserting it.
# (sRGB->XYZ matrix, D65 white point, and CIELAB constants live in
# tramage._constants and are imported under these names at the top of the file.)


def _lab_f(t: FloatArray) -> FloatArray:
    """CIELAB nonlinear function on normalized XYZ ratios."""
    return np.where(t > _EPS_LAB, np.cbrt(t), _KAPPA_LAB * t + 4.0 / 29.0)


def _lab_f_inv(t: FloatArray) -> FloatArray:
    """Inverse of :func:`_lab_f`, mapping CIELAB back to normalized XYZ ratios."""
    t3 = t ** 3
    # Linear branch coefficient is 3*(6/29)^2 (NOT multiplied by eps; eps is
    # the forward threshold). See CIE 15:2004, or Bruce Lindbloom's reference.
    return np.where(t > 6.0 / 29.0, t3, 3.0 * (6.0 / 29.0) ** 2 * (t - 4.0 / 29.0))


def srgb_to_cielab(rgb: FloatArray) -> FloatArray:
    """Convert gamma-encoded sRGB in [0, 1] to CIELAB (D65).

    Returns ``(L*, a*, b*)`` with L* on the standard 0-100 scale. Uses the
    IEC 61966-2-1 sRGB-to-XYZ matrix and the standard CIELAB cube-root
    nonlinearity with the 6/29 threshold.
    """
    lin = srgb_to_linear(rgb)
    xyz = lin @ _SRGB_TO_XYZ_D65.T
    fx = _lab_f(xyz[..., 0] / _XN)
    fy = _lab_f(xyz[..., 1] / _YN)
    fz = _lab_f(xyz[..., 2] / _ZN)
    return np.stack([116.0 * fy - 16.0, 500.0 * (fx - fy), 200.0 * (fy - fz)], axis=-1)


def cielab_to_srgb(lab: FloatArray) -> FloatArray:
    """Convert CIELAB (D65) back to gamma-encoded sRGB.

    Same clipping convention as :func:`oklab_to_srgb`: negative
    linear-light values are clipped to zero, while values above 1 pass
    through, so out-of-gamut colors come back as out-of-range floats on
    the high side. For out-of-gamut detection use :func:`in_gamut`
    (Oklab input), which checks linear light before the transfer function.
    """
    fy = (lab[..., 0] + 16.0) / 116.0
    fx = lab[..., 1] / 500.0 + fy
    fz = fy - lab[..., 2] / 200.0
    xyz = np.stack([_XN * _lab_f_inv(fx), _YN * _lab_f_inv(fy), _ZN * _lab_f_inv(fz)], axis=-1)
    lin = xyz @ _XYZ_FROM_SRGB_INV.T
    return linear_to_srgb(lin)


def srgb_to_xyz(rgb: FloatArray) -> FloatArray:
    """Convert linear sRGB in [0, 1] to CIE XYZ (D65, 2-degree observer).

    Uses the IEC 61966-2-1 sRGB-to-XYZ matrix without any intermediate
    transfer function: the caller must supply linear-light sRGB values,
    for example from :func:`srgb_to_linear`.
    """
    rgb = np.asarray(rgb, dtype=np.float64)
    return rgb @ _SRGB_TO_XYZ_D65.T


def oklab_to_xyz(lab: FloatArray) -> FloatArray:
    """Convert Oklab to CIE XYZ (D65, 2-degree observer).

    Round-trips through linear sRGB using :func:`oklab_to_srgb` and
    :func:`srgb_to_linear`, then applies the standard sRGB-to-XYZ matrix.
    Provided so that metrics and experiments that need XYZ (for example,
    S-CIELAB) can reach it from the package's native Oklab without
    duplicating conversion constants.
    """
    return srgb_to_xyz(srgb_to_linear(oklab_to_srgb(lab)))
