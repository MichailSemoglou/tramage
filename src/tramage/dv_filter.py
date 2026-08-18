"""Matrix-valued error-diffusion filter design per Damera-Venkata & Evans (2001).

Damera-Venkata and Evans generalise the scalar Kolpatzik-Bouman error
filters to full matrix-valued taps: each Floyd-Steinberg tap is a 3×3
matrix rather than a scalar, allowing cross-channel coupling that a
channel-separable filter cannot express. The filter is designed to
minimise perceived error variance under the Poirson-Wandell
pattern-colour separable human vision model, by solving the matrix
Yule-Walker equations.

Reference:
  N. Damera-Venkata and B. L. Evans, "Design and analysis of vector color
  error diffusion halftoning systems," IEEE Trans. Image Processing,
  vol. 10, no. 10, pp. 1552–1565, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import gaussian_filter

from ._constants import (
    OPP_TO_XYZ,
    SIGMA_CHROM_DEG,
    SIGMA_LUM_DEG,
    XYZ_TO_LINEAR_SRGB,
    XYZ_TO_OPP,
    pixels_per_degree,
)
from .colour import FloatArray, linear_to_srgb, oklab_to_xyz, srgb_to_oklab
from .diffusion import _scanlines, nearest_palette_index

_FS_POSITIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, -1), (1, 0), (1, 1))
_N_TAPS: int = len(_FS_POSITIONS)


@dataclass
class DVFilters:
    """Matrix-valued error-diffusion filter taps.

    Attributes
    ----------
    weights:
        Shape ``(4, 3, 3)`` — one 3×3 matrix per FS tap position.
        ``weights[k]`` is the tap at ``((0,1), (1,-1), (1,0), (1,1))[k]``.
    K:
        Shape ``(3, 3)`` — the linear quantiser gain matrix estimated from
        the palette. ``K ≈ I`` for a broad palette; ``K`` shrinks toward
        zero along directions the palette barely represents.
    display_dpi:
        Pixels per inch the filter was designed for.
    viewing_distance_cm:
        Viewing distance in centimetres the filter was designed for.
    """

    weights: np.ndarray
    K: np.ndarray
    display_dpi: float
    viewing_distance_cm: float


def _oklab_to_opponent(oklab: FloatArray) -> FloatArray:
    """Oklab -> linear sRGB -> XYZ D65 -> Poirson-Wandell opponent."""
    xyz = oklab_to_xyz(oklab)
    return xyz @ XYZ_TO_OPP.T


def _opponent_to_oklab(opp: FloatArray) -> FloatArray:
    """Poirson-Wandell opponent -> XYZ D65 -> linear sRGB -> Oklab."""
    xyz = opp @ OPP_TO_XYZ.T
    lin_srgb = xyz @ XYZ_TO_LINEAR_SRGB.T
    return srgb_to_oklab(linear_to_srgb(lin_srgb))


def _nearest_palette_vec(pixels: FloatArray, palette: FloatArray) -> np.ndarray:
    """Vectorised nearest-palette-index for shape ``(n, 3)``."""
    return np.argmin(((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1), axis=-1)


def _hvs_filter(
    error: FloatArray,
    sigma_lum: float,
    sigma_chrom: float,
) -> FloatArray:
    """Apply channel-specific Gaussian CSF filters to the error in opponent space.

    Luminance channel (O1): narrower Gaussian (sigma_lum).
    Chrominance channels (O2, O3): wider Gaussian (sigma_chrom).
    """
    return np.stack([
        gaussian_filter(error[..., 0], sigma_lum),
        gaussian_filter(error[..., 1], sigma_chrom),
        gaussian_filter(error[..., 2], sigma_chrom),
    ], axis=-1)


def _lagged_correlation(
    a: FloatArray,
    b: FloatArray,
    dy: int,
    dx: int,
) -> np.ndarray:
    """Estimate E[a[y,x] @ b[y-dy, x-dx]^T] from two ``(H, W, 3)`` fields.

    Passing the same field for both ``a`` and ``b`` yields its
    autocorrelation at lag ``(dy, dx)``; passing a filtered and an unfiltered
    field yields their cross-correlation.
    """
    if dy < 0:
        a = a[:dy, :]
        b = b[-dy:, :]
    elif dy > 0:
        a = a[dy:, :]
        b = b[:-dy, :]
    if dx < 0:
        a = a[:, :dx]
        b = b[:, -dx:]
    elif dx > 0:
        a = a[:, dx:]
        b = b[:, :-dx]
    a_flat = a.reshape(-1, 3)
    b_flat = b.reshape(-1, 3)
    return (a_flat.T @ b_flat) / a_flat.shape[0]


def _design_dv_taps(
    pal_opp: FloatArray,
    ppd: float,
    sigma_lum: float,
    sigma_chrom: float,
    K: np.ndarray,
    rng: np.random.Generator,
    train_size: int = 128,
    n_iter: int = 3,
) -> tuple[np.ndarray, list[float]]:
    """Design matrix-valued filter taps by iterated Yule-Walker solve.

    Each iteration: run error diffusion with the current filter on a
    random training patch, estimate error autocorrelations, solve the
    matrix Yule-Walker system, and update the taps. The DC constraint
    (sum of taps = identity) is enforced by projection after each solve.

    Parameters
    ----------
    pal_opp:
        Palette in opponent space, shape ``(n, 3)``.
    ppd:
        Pixels per degree of visual angle.
    sigma_lum, sigma_chrom:
        Gaussian CSF sigmas in pixel units.
    K:
        Linear quantiser gain matrix, shape ``(3, 3)``.
    rng:
        Seeded random generator.
    train_size:
        Width/height of the square training patch.
    n_iter:
        Number of Yule-Walker solve iterations.

    Returns
    -------
    ``(weights, loss_values)``: weights has shape ``(4, 3, 3)``.
    """
    weights = np.array([np.eye(3) / _N_TAPS for _ in range(_N_TAPS)])
    losses: list[float] = []

    pal_min = pal_opp.min(axis=0) - 0.05
    pal_max = pal_opp.max(axis=0) + 0.05

    for _ in range(n_iter):
        train = rng.uniform(
            pal_min, pal_max, size=(train_size, train_size, 3)
        )

        error = _run_diffusion_error(train, pal_opp, weights)
        e_w = _hvs_filter(error, sigma_lum, sigma_chrom)

        mse = float((e_w ** 2).mean())
        losses.append(mse)

        M, rhs = _build_yule_walker_system(error, e_w, _FS_POSITIONS)
        if np.any(~np.isfinite(M)) or np.any(~np.isfinite(rhs)):
            losses.append(float("nan"))
            break
        h_flat = np.linalg.lstsq(M, rhs, rcond=None)[0]

        weights_new = h_flat.reshape(_N_TAPS, 3, 3)
        correction = (weights_new.sum(axis=0) - np.eye(3)) / _N_TAPS
        weights = weights_new - correction

    return weights, losses


def _build_yule_walker_system(
    error: FloatArray,
    e_w: FloatArray,
    positions: tuple[tuple[int, int], ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Assemble the matrix Yule-Walker system for optimal filter taps.

    Minimizes E[||W*e_final||^2] where
        e_final = e - sum_j h_j @ e_delayed_j.

    The orthogonality principle gives, for each tap k:
        sum_j h_j @ R(p_j - p_k) = C_W_k

    where R(Δ) = E[e[n] @ e[n-Δ]^T] and
          C_W_k = E[e_W[n] @ e[n-p_k]^T].

    The system is assembled as a 36×36 matrix using Kronecker products:
        block(k, j) = R(p_j - p_k)^T ⊗ I_3.
    """
    n_taps = len(positions)
    block_size = 9
    M = np.zeros((block_size * n_taps, block_size * n_taps))
    rhs_vec = np.zeros(block_size * n_taps)

    for k_idx in range(n_taps):
        pk_y, pk_x = positions[k_idx]
        Ck = _lagged_correlation(e_w, error, pk_y, pk_x)
        rhs_vec[block_size * k_idx: block_size * (k_idx + 1)] = Ck.ravel(order="F")

        for j_idx in range(n_taps):
            pj_y, pj_x = positions[j_idx]
            dy = pj_y - pk_y
            dx = pj_x - pk_x
            R_mat = _lagged_correlation(error, error, dy, dx)
            block = np.kron(R_mat.T, np.eye(3))
            M[block_size * k_idx: block_size * (k_idx + 1),
              block_size * j_idx: block_size * (j_idx + 1)] = block

    return M, rhs_vec


def _run_diffusion_error(
    image_opp: FloatArray,
    palette_opp: FloatArray,
    weights: np.ndarray,
) -> FloatArray:
    """Run matrix-valued error diffusion and return the per-pixel error signal.

    Parameters
    ----------
    image_opp:
        Training image in opponent space, shape ``(H, W, 3)``.
    palette_opp:
        Palette in opponent space, shape ``(n, 3)``.
    weights:
        Filter taps, shape ``(4, 3, 3)``.

    Returns
    -------
    Error field ``(H, W, 3)``: image + feedback - quantised output.
    """
    height, width = image_opp.shape[:2]
    buf = image_opp.copy()
    error = np.zeros_like(image_opp)

    for y, xs, sign in _scanlines(height, width, serpentine=True):
        for x in xs:
            pixel = buf[y, x]
            index = nearest_palette_index(pixel, palette_opp)
            e = pixel - palette_opp[index]
            e = np.clip(e, -1e3, 1e3)
            error[y, x] = e
            for k, (dy, dx) in enumerate(_FS_POSITIONS):
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny, nx] += weights[k] @ e
                    buf[ny, nx] = np.clip(buf[ny, nx], -1e6, 1e6)

    return error


def dv_filters(
    palette_oklab: FloatArray,
    display_dpi: float = 100.0,
    viewing_distance_cm: float = 40.0,
    *,
    train_samples: int = 5000,
    train_size: int = 128,
    seed: int = 42,
) -> DVFilters:
    """Design matrix-valued error-diffusion filters for a palette.

    The filter is designed in Poirson-Wandell opponent colour space with
    channel-specific CSF Gaussian filters. Each FS-position tap is a
    3×3 matrix, allowing cross-channel coupling.

    Parameters
    ----------
    palette_oklab:
        Palette in Oklab, shape ``(n, 3)``.
    display_dpi:
        Pixels per inch. Together with `viewing_distance_cm` determines
        the visual-angle extent of the CSF Gaussians in pixel units.
    viewing_distance_cm:
        Typical viewing distance in centimetres.
    train_samples:
        Number of random colours used to estimate the quantiser gain K.
    train_size:
        Square training patch width/height for the Yule-Walker solve.
    seed:
        Seed for reproducible filter design.

    Returns
    -------
    :class:`DVFilters` with the designed taps and quantiser gain matrix.
    """
    rng = np.random.default_rng(seed)
    palette = np.asarray(palette_oklab, dtype=np.float64)
    if len(palette) < 2:
        raise ValueError(f"dv_filters requires at least 2 palette colours, got {len(palette)}")

    ppd = pixels_per_degree(display_dpi, viewing_distance_cm)
    sigma_lum = SIGMA_LUM_DEG * ppd
    sigma_chrom = SIGMA_CHROM_DEG * ppd

    pal_opp = _oklab_to_opponent(palette)

    # --- estimate quantiser gain K in opponent space ---
    pal_min = palette.min(axis=0) - 0.05
    pal_max = palette.max(axis=0) + 0.05
    train = rng.uniform(pal_min, pal_max, size=(train_samples, 3))
    train_opp = _oklab_to_opponent(train)
    indices = _nearest_palette_vec(train_opp, pal_opp)
    train_q_opp = pal_opp[indices]

    x = train_opp - train_opp.mean(axis=0)
    x_q = train_q_opp - train_q_opp.mean(axis=0)
    C_x = (x.T @ x) / train_samples
    C_xq = (x_q.T @ x) / train_samples
    K = C_xq @ np.linalg.inv(C_x)

    # --- design filter taps ---
    weights, _ = _design_dv_taps(
        pal_opp=pal_opp,
        ppd=ppd,
        sigma_lum=sigma_lum,
        sigma_chrom=sigma_chrom,
        K=K,
        rng=rng,
        train_size=train_size,
    )

    return DVFilters(
        weights=weights,
        K=K,
        display_dpi=display_dpi,
        viewing_distance_cm=viewing_distance_cm,
    )


def dv_diffuse(
    image_oklab: FloatArray,
    palette_oklab: FloatArray,
    filters: DVFilters | None = None,
    shaping: FloatArray | None = None,
    serpentine: bool = True,
) -> tuple[FloatArray, np.ndarray]:
    """Quantise ``image`` to ``palette`` using matrix-valued DV error diffusion.

    Diffuses in Poirson-Wandell opponent space with matrix-valued filter
    taps. If ``filters`` is ``None``, designs a filter for the given
    palette at the default 100 dpi / 40 cm viewing geometry.

    ``shaping``, if given, must be a single ``(3, 3)`` global shaping
    matrix applied in Oklab before error is distributed. Per-cell
    matrices (``(n, 3, 3)``) are not supported by this function. Use
    :func:`tramage.diffusion.diffuse` for per-cell shaping.

    Parameters
    ----------
    image_oklab:
        Continuous-tone image in Oklab, shape ``(H, W, 3)``.
    palette_oklab:
        Palette in Oklab, shape ``(n, 3)``.
    filters:
        Pre-designed :class:`DVFilters`, or ``None`` to design on the fly.
    shaping:
        Optional single ``(3, 3)`` global shaping matrix.
    serpentine:
        Alternate scan direction each row.

    Returns
    -------
    ``(dithered, indices)``: the quantised image in Oklab and the
    per-pixel palette index.
    """
    image_oklab = np.asarray(image_oklab, dtype=np.float64)
    palette_oklab = np.asarray(palette_oklab, dtype=np.float64)

    if filters is None:
        filters = dv_filters(palette_oklab)

    if shaping is not None:
        shaping = np.asarray(shaping, dtype=np.float64)
        if shaping.ndim != 2:
            raise ValueError(
                f"dv_diffuse only supports global shaping (a single (3, 3) matrix), "
                f"got shape {shaping.shape}. Use tramage.diffusion.diffuse for "
                f"per-cell shaping."
            )

    # Convert image and palette to opponent space for the diffusion loop.
    image_opp = _oklab_to_opponent(image_oklab)
    pal_opp = _oklab_to_opponent(palette_oklab)
    weights = filters.weights

    height, width = image_opp.shape[:2]
    buf = image_opp.copy()
    indices = np.zeros((height, width), dtype=np.int64)

    for y, xs, sign in _scanlines(height, width, serpentine):
        for x in xs:
            pixel = buf[y, x]
            index = nearest_palette_index(pixel, pal_opp)
            indices[y, x] = index
            error = pixel - pal_opp[index]

            if shaping is not None:
                err_ok = _opponent_to_oklab(error)
                err_ok = shaping @ err_ok
                error = _oklab_to_opponent(err_ok)

            for k, (dy, dx) in enumerate(_FS_POSITIONS):
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny, nx] += weights[k] @ error

    dithered_opp = pal_opp[indices]
    return _opponent_to_oklab(dithered_opp), indices
