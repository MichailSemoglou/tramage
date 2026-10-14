"""Metrics for judging dithered output against the continuous-tone original.

These quantify the fidelity-versus-noise trade-off that palette shaping is
built around: how much systematic bias remains, how blurred tone compares
to the original, how noisy each channel is, and whether that noise is
concentrated at low spatial frequencies (visible as "worms") or spread
across the spectrum in the diffuse pattern usually called blue noise.

All functions take the dithered image and the original continuous-tone
image, both in Oklab, shape ``(H, W, 3)``.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import fftconvolve

from ._constants import (
    D65_XN,
    D65_YN,
    D65_ZN,
    OPP_TO_XYZ,
    SCIELAB_BY_SPREADS_DEG,
    SCIELAB_BY_WEIGHTS,
    SCIELAB_LUM_SPREADS_DEG,
    SCIELAB_LUM_WEIGHTS,
    SCIELAB_RG_SPREADS_DEG,
    SCIELAB_RG_WEIGHTS,
    SIGMA_CHROM_DEG,
    SIGMA_LUM_DEG,
    XYZ_TO_OPP,
    pixels_per_degree,
)
from .colour import FloatArray, _lab_f, oklab_to_xyz


def error(dithered: FloatArray, original: FloatArray) -> FloatArray:
    """Per-pixel error vector, dithered minus original."""
    return dithered - original


def systematic_bias(dithered: FloatArray, original: FloatArray) -> FloatArray:
    """Mean error over the image, one value per channel.

    A shaping matrix that only redistributes error within the palette's
    affine hull cannot change this: it is the part of the error that no
    amount of local redistribution removes.
    """
    return error(dithered, original).mean(axis=(0, 1))


def noise_std(dithered: FloatArray, original: FloatArray) -> FloatArray:
    """Per-channel standard deviation of the error, one value per channel."""
    return error(dithered, original).std(axis=(0, 1))


def blurred_tone_error(
    dithered: FloatArray,
    original: FloatArray,
    sigma: float = 1.5,
) -> float:
    """Mean Oklab distance between Gaussian-blurred versions of both images.

    Blurring approximates how the eye averages a dither pattern at normal
    viewing distance, so this measures whether the *tone* the dither
    reproduces (rather than the pixel-exact error) matches the original.

    Parameters
    ----------
    dithered:
        Dithered output in Oklab, shape ``(H, W, 3)``.
    original:
        Continuous-tone original in Oklab, shape ``(H, W, 3)``.
    sigma:
        Standard deviation of the Gaussian blur kernel, in pixels
        (default 1.5 px).

    """
    channels = dithered.shape[-1]
    blurred_dithered = np.stack(
        [gaussian_filter(dithered[..., c], sigma) for c in range(channels)], axis=-1
    )
    blurred_original = np.stack(
        [gaussian_filter(original[..., c], sigma) for c in range(channels)], axis=-1
    )
    return float(np.linalg.norm(blurred_dithered - blurred_original, axis=-1).mean())


def lowfreq_energy_fraction(
    dithered: FloatArray,
    original: FloatArray,
    channel: int,
    radius_frac: float = 0.10,
) -> float:
    """Fraction of a channel's error power concentrated at low frequency.

    Error diffusion that cannot shed unpayable error tends to push it into
    slow, low-frequency drift ("worms") rather than the high-frequency blue
    noise that is visually inert at normal viewing distance. This returns
    the share of spectral power inside a disc of radius `radius_frac` times
    the Nyquist frequency, excluding the DC term.

    Parameters
    ----------
    dithered:
        Dithered output in Oklab, shape ``(H, W, 3)``.
    original:
        Continuous-tone original in Oklab, shape ``(H, W, 3)``.
    channel:
        Channel index (0 = L, 1 = a, 2 = b) whose error spectrum is
        measured.
    radius_frac:
        Radius of the low-frequency disc as a fraction of the Nyquist
        frequency (default 0.10).

    Raises
    ------
    ValueError
        If `channel` is out of range for the image's channel count.

    """
    if not 0 <= channel < dithered.shape[-1]:
        raise ValueError(
            f"channel must be in [0, {dithered.shape[-1]}), got {channel}"
        )
    e = error(dithered, original)[..., channel]
    e = e - e.mean()
    height, width = e.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
    cy, cx = height // 2, width // 2
    radius = np.hypot(*np.mgrid[-cy:height - cy, -cx:width - cx])
    power[cy, cx] = 0
    nyquist = min(height, width) / 2
    band = (radius > 0) & (radius < radius_frac * nyquist)
    total = power[radius > 0].sum()
    if total == 0:
        return 0.0
    return float(power[band].sum() / total)


def spectral_anisotropy(
    dithered: FloatArray,
    original: FloatArray,
    channel: int,
    band: tuple[float, float] = (0.15, 0.45),
    bins: int = 18,
) -> float:
    """Angular non-uniformity of a channel's error spectrum in a frequency band.

    A well-behaved diffusion pattern spreads error roughly evenly across
    orientation within the mid-frequency band; directional artifacts (from
    serpentine scanning bias, for example) show up as an uneven angular
    profile. Returns the coefficient of variation of spectral power across
    `bins` angular bins: 0 means every orientation carries equal power, and 1
    means the angular standard deviation equals the angular mean (strongly
    directional). Values can exceed 1 for highly peaked spectra; the scale is
    relative, not normalized to a fixed maximum.

    Parameters
    ----------
    dithered:
        Dithered output in Oklab, shape ``(H, W, 3)``.
    original:
        Continuous-tone original in Oklab, shape ``(H, W, 3)``.
    channel:
        Channel index (0 = L, 1 = a, 2 = b) whose error spectrum is
        measured.
    band:
        ``(low, high)`` radial-frequency limits of the measured band, as
        fractions of the Nyquist frequency (default ``(0.15, 0.45)``).
    bins:
        Number of angular bins the band is divided into (default 18).

    Raises
    ------
    ValueError
        If `channel` is out of range for the image's channel count.

    """
    if not 0 <= channel < dithered.shape[-1]:
        raise ValueError(
            f"channel must be in [0, {dithered.shape[-1]}), got {channel}"
        )
    e = error(dithered, original)[..., channel]
    e = e - e.mean()
    height, width = e.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(e))) ** 2
    cy, cx = height // 2, width // 2
    gy, gx = np.mgrid[-cy:height - cy, -cx:width - cx]
    radius = np.hypot(gy, gx)
    theta = np.arctan2(gy, gx) % np.pi
    nyquist = min(height, width) / 2
    in_band = (radius > band[0] * nyquist) & (radius < band[1] * nyquist)
    power_sum, _ = np.histogram(theta[in_band], bins=bins, weights=power[in_band])
    counts, _ = np.histogram(theta[in_band], bins=bins)
    profile = power_sum / np.maximum(counts, 1)
    if profile.mean() == 0:
        return 0.0
    return float(profile.std() / profile.mean())


# --- S-CIELAB (Zhang & Wandell 1997) ----------------------------------------
#
# S-CIELAB applies spatial pre-filtering in an opponent color space before
# computing CIELAB ΔE, so that the metric accounts for the human visual
# system's reduced sensitivity to high-spatial-frequency color variations.
# This is the standard perceptual metric for halftoning and error-diffusion
# comparisons (Damera-Venkata & Evans 2001, Kite-Evans-Bovik 2000).
#
# The implementation uses:
#   - The Poirson & Wandell (1996) pattern-color separable opponent matrix
#     to convert XYZ → opponent (luminance O1, red-green O2, blue-yellow O3).
#   - The tabulated S-CIELAB spatial filters (Zhang & Wandell 1997,
#     Appendix A): per opponent channel, a weighted sum of unit-volume 2D
#     Gaussians with the published spreads, each lobe normalized to unit
#     sum and the composite normalized to unit DC gain. The luminance
#     filter's third lobe carries a negative weight, giving the bandpass
#     surround of the luminance CSF.
#   - Standard CIELAB ΔE (CIE76) on the filtered images.
#
# A single-Gaussian approximation of the filters (luminance: the narrowest
# published lobe only; chrominance: one shared spread, dropping the
# red-green/blue-yellow distinction) is retained as ``filters="gaussian"``
# for comparison; it is not the metric the paper reports.

# (Opponent matrix, D65 white point, CIELAB constants, and the filter
# parameters live in tramage._constants and are imported at the top of the
# file.)


def _xyz_to_opponent(xyz: FloatArray) -> FloatArray:
    """XYZ D65 → opponent (O1 luminance, O2 rg, O3 yb)."""
    return xyz @ XYZ_TO_OPP.T


def _opponent_to_xyz(opp: FloatArray) -> FloatArray:
    """Opponent → XYZ D65."""
    return opp @ OPP_TO_XYZ.T


def _xyz_to_cielab_per_pixel(xyz: FloatArray) -> FloatArray:
    """XYZ D65 → CIELAB L*a*b*, reusing colour._lab_f for the nonlinearity."""
    fx = _lab_f(xyz[..., 0] / D65_XN)
    fy = _lab_f(xyz[..., 1] / D65_YN)
    fz = _lab_f(xyz[..., 2] / D65_ZN)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def _gaussian_lobe(sigma_px: float, half: int) -> FloatArray:
    """Unit-sum 2D Gaussian of pixel sigma `sigma_px` on a (2*half+1)² grid."""
    ax = np.arange(-half, half + 1, dtype=np.float64)
    yy, xx = np.meshgrid(ax, ax, indexing="ij")
    lobe = np.exp(-(yy**2 + xx**2) / (2.0 * sigma_px**2))
    return lobe / lobe.sum()


@lru_cache(maxsize=8)
def _scielab_kernels(dpi: float, viewing_distance_cm: float) -> tuple[FloatArray, ...]:
    """The three tabulated S-CIELAB opponent filters as 2D kernels.

    Built from the published weights and spreads (Zhang & Wandell 1997,
    Appendix A) sampled on the pixel grid implied by the viewing geometry.
    Support is ±4σ of each channel's widest lobe. Cached: the kernels
    depend only on the viewing geometry, not on the image.
    """
    ppd = pixels_per_degree(dpi, viewing_distance_cm)
    kernels = []
    for weights, spreads_deg in (
        (SCIELAB_LUM_WEIGHTS, SCIELAB_LUM_SPREADS_DEG),
        (SCIELAB_RG_WEIGHTS, SCIELAB_RG_SPREADS_DEG),
        (SCIELAB_BY_WEIGHTS, SCIELAB_BY_SPREADS_DEG),
    ):
        sigmas_px = [spread * ppd for spread in spreads_deg]
        half = int(math.ceil(4.0 * max(sigmas_px)))
        kernel = np.zeros((2 * half + 1, 2 * half + 1), dtype=np.float64)
        for weight, sigma_px in zip(weights, sigmas_px):
            kernel += weight * _gaussian_lobe(sigma_px, half)
        kernel /= kernel.sum()
        kernels.append(kernel)
    return tuple(kernels)


def _convolve_symmetric(image: FloatArray, kernel: FloatArray) -> FloatArray:
    """Convolve a 2D image with `kernel`, symmetric (edge-repeating) borders.

    This is the ``mode="reflect"`` convention of :func:`scipy.ndimage.
    gaussian_filter` (which the Gaussian-approximation path uses), computed
    by FFT: the luminance kernel's wide surround (σ ≈ 119 px at the default
    geometry) makes the separable direct form needlessly slow.
    """
    pad_h, pad_w = kernel.shape[0] // 2, kernel.shape[1] // 2
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode="symmetric")
    return fftconvolve(padded, kernel, mode="valid")


def _filter_opponent(
    opp: FloatArray,
    dpi: float,
    viewing_distance_cm: float,
    filters: str,
) -> FloatArray:
    """Apply the per-channel S-CIELAB spatial pre-filters to an opponent image."""
    if filters == "zhang1997":
        kernels = _scielab_kernels(dpi, viewing_distance_cm)
        return np.stack(
            [_convolve_symmetric(opp[..., c], kernels[c]) for c in range(3)],
            axis=-1,
        )
    if filters == "gaussian":
        ppd = pixels_per_degree(dpi, viewing_distance_cm)
        sigma_lum_px = SIGMA_LUM_DEG * ppd
        sigma_chrom_px = SIGMA_CHROM_DEG * ppd
        return np.stack([
            gaussian_filter(opp[..., 0], sigma_lum_px),
            gaussian_filter(opp[..., 1], sigma_chrom_px),
            gaussian_filter(opp[..., 2], sigma_chrom_px),
        ], axis=-1)
    raise ValueError(
        f"unknown filters {filters!r}: expected 'zhang1997' or 'gaussian'"
    )


def scielab_de(
    dithered: FloatArray,
    original: FloatArray,
    dpi: float = 100.0,
    viewing_distance_cm: float = 40.0,
    *,
    full: bool = False,
    filters: str = "zhang1997",
) -> float | dict[str, float]:
    """S-CIELAB ΔE between dithered and original images (both in Oklab).

    Applies the opponent-space spatial pre-filtering of Zhang & Wandell
    (1997), the human contrast sensitivity functions per opponent channel,
    before computing standard CIELAB ΔE (CIE76). This is the metric the
    halftoning literature trusts for perceptual image quality: it penalizes
    visible structured error while discounting high-frequency noise the eye
    cannot resolve.

    Parameters
    ----------
    dithered:
        Dithered output in Oklab, shape ``(H, W, 3)``.
    original:
        Continuous-tone original in Oklab, shape ``(H, W, 3)``.
    dpi:
        Pixels per inch of the display the image is viewed on. Together
        with `viewing_distance_cm` this determines the spatial extent of
        the contrast sensitivity filters in pixel units.
    viewing_distance_cm:
        Typical viewing distance in centimeters. The defaults (100 DPI,
        40 cm) place 27.5 pixels per degree of visual angle. The
        Kolpatzik-Bouman filters in :mod:`tramage.kb_filter` were designed
        at a 0.0317°/pixel sampling rate (31.5 pixels per degree at 40 cm,
        ≈115 DPI); this geometry is ~13% coarser, stated here rather than
        silently matched.
    full:
        If ``False`` (default), returns the mean ΔE across all pixels.
        If ``True``, returns a dict with ``"mean"``, ``"p95"``, and
        ``"max"``.
    filters:
        ``"zhang1997"`` (default) applies the tabulated S-CIELAB kernels:
        weighted sums of the published Gaussians per opponent channel,
        including the luminance bandpass surround. ``"gaussian"`` applies
        the single-Gaussian approximation of the S-CIELAB kernels;
        retained for comparison, not for reporting.

    Returns
    -------
    ``float`` or ``dict``
        Mean S-CIELAB ΔE, or summary dict if ``full=True``.

    Raises
    ------
    ValueError
        If `filters` is neither ``"zhang1997"`` nor ``"gaussian"``.

    Notes
    -----
    The two ``filters`` paths share the opponent transform, viewing
    geometry, and CIE76 differencing; they differ only in the spatial
    kernels.

    """
    xyz_dithered = oklab_to_xyz(dithered)
    xyz_original = oklab_to_xyz(original)

    opp_dithered = _xyz_to_opponent(xyz_dithered)
    opp_original = _xyz_to_opponent(xyz_original)

    filtered_dithered = _filter_opponent(opp_dithered, dpi, viewing_distance_cm, filters)
    filtered_original = _filter_opponent(opp_original, dpi, viewing_distance_cm, filters)

    lab_dithered = _xyz_to_cielab_per_pixel(_opponent_to_xyz(filtered_dithered))
    lab_original = _xyz_to_cielab_per_pixel(_opponent_to_xyz(filtered_original))

    de_map = np.linalg.norm(lab_dithered - lab_original, axis=-1)
    de_mean = float(de_map.mean())
    if not full:
        return de_mean
    de_flat = np.sort(de_map.ravel())
    return {
        "mean": de_mean,
        "p95": float(de_flat[int(len(de_flat) * 0.95)]),
        "max": float(de_flat[-1]),
    }
