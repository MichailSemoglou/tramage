"""Matrix-valued error-diffusion filter design per Damera-Venkata & Evans (2001).

Damera-Venkata and Evans generalize the scalar Kolpatzik-Bouman error
filters to full matrix-valued taps: each Floyd-Steinberg tap is a 3×3
matrix rather than a scalar, allowing cross-channel coupling that a
channel-separable filter cannot express. The filter is designed to
minimize perceived error variance under the Poirson-Wandell
pattern-color separable human vision model, by solving the matrix
Yule-Walker equations.

Two properties of this implementation are worth stating, because they
differ from a naive reading of the paper:

1. The design is **direct, not trained**. Under DV&E's white-noise model
   of the quantizer residual (their eqs. 64-65), the generalized
   Yule-Walker system (their eq. 63) involves only the autocorrelation
   of the HVS filter's matrix impulse response, and no training images.

2. The diffusion runs **in Oklab**, the metric space. The opponent
   transform enters only through the design, via G, the Jacobian of the
   Oklab→(Poirson-Wandell opponent) map at the palette centroid, so the
   matrix impulse response W[n] = diag(k_c[n]) @ G maps Oklab error
   vectors to perceived opponent space. Because G is a fixed linear map,
   running the diffusion in Oklab is what makes the taps' DC conservation
   (sum h_j = I) protect the metric space's mean: diffusing in a working
   space that is a *nonlinear* function of Oklab would conserve that
   space's mean instead, leaking image-correlated low-frequency error.

The DC constraint (sum of taps = I, DV&E's "lossless diffusion"
constraint, their eq. 43) is imposed exactly, via Lagrange multipliers,
rather than by projection. The designed filters are aggressive (they
whiten the perceived noise), which puts the real quantizer loop at the
edge of overload; ``dv_diffuse`` therefore clamps the accumulation
buffer to a small margin around the palette's bounding box, the
standard practical stabilizer for error diffusion with strong filters.

Reference:
  N. Damera-Venkata and B. L. Evans, "Design and analysis of vector color
  error diffusion halftoning systems," IEEE Trans. Image Processing,
  vol. 10, no. 10, pp. 1552–1565, 2001.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ._constants import OPP_TO_XYZ, XYZ_TO_LINEAR_SRGB, XYZ_TO_OPP
from .colour import FloatArray, linear_to_srgb, oklab_to_xyz, srgb_to_oklab
from .diffusion import _scanlines, nearest_palette_index
from .metrics import _scielab_kernels

_FS_POSITIONS: tuple[tuple[int, int], ...] = ((0, 1), (1, -1), (1, 0), (1, 1))
_N_TAPS: int = len(_FS_POSITIONS)

# Lags at which the HVS impulse-response autocorrelation is needed: the tap
# positions themselves (right-hand side) and all pairwise differences.
_DESIGN_LAGS: tuple[tuple[int, int], ...] = tuple(
    sorted(
        set(_FS_POSITIONS)
        | {(pk[0] - pj[0], pk[1] - pj[1]) for pj in _FS_POSITIONS for pk in _FS_POSITIONS}
    )
)

# Margin outside the palette's Oklab bounding box within which the
# accumulation buffer is allowed to drift in ``dv_diffuse``. Bounds the loop
# state against quantizer overload with aggressive HVS-optimal filters.
_CLAMP_MARGIN: float = 0.1

# Scalar Floyd-Steinberg weights as (dy, dx, weight), used only as the
# fallback filter when the opponent linearization is degenerate.
_FS_WEIGHTS_FALLBACK: tuple[tuple[int, int, float], ...] = (
    (0, 1, 7 / 16),
    (1, -1, 3 / 16),
    (1, 0, 5 / 16),
    (1, 1, 1 / 16),
)


@dataclass
class DVFilters:
    """Matrix-valued error-diffusion filter taps.

    Attributes
    ----------
    weights:
        Shape ``(4, 3, 3)``: one 3×3 matrix per FS tap position.
        ``weights[k]`` is the tap at ``((0,1), (1,-1), (1,0), (1,1))[k]``.
        The taps act on Oklab error vectors and always sum to the identity
        (the DC or "lossless diffusion" constraint).
    K:
        Shape ``(3, 3)``: the linear quantizer gain matrix estimated from
        the palette in Oklab. ``K ≈ I`` for a broad palette; ``K`` shrinks
        toward zero along directions the palette barely represents.
        Informational: the direct Yule-Walker design does not use it (the
        white-noise covariance cancels out of the normal equations).
    display_dpi:
        Pixels per inch the filter was designed for.
    viewing_distance_cm:
        Viewing distance in centimeters the filter was designed for.

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
    """Vectorized nearest-palette-index for shape ``(n, 3)``."""
    return np.argmin(((pixels[:, None, :] - palette[None, :, :]) ** 2).sum(axis=-1), axis=-1)


def _opponent_jacobian(centre: FloatArray) -> np.ndarray:
    """Jacobian of :func:`_oklab_to_opponent` at ``centre``, shape (3, 3).

    The opponent transform is a nonlinear function of Oklab (through the
    Oklab→XYZ cube); this is its local linearization, mapping small Oklab
    error vectors to opponent-space error vectors. Central differences with
    a fixed step; the map is smooth and locally cubic, so the truncation
    error is O(1e-10) at the step used.
    """
    centre = np.asarray(centre, dtype=np.float64)
    eps = 1e-5
    jac = np.zeros((3, 3))
    for i in range(3):
        step = np.zeros(3)
        step[i] = eps
        jac[:, i] = (_oklab_to_opponent(centre + step) - _oklab_to_opponent(centre - step)) / (
            2 * eps
        )
    return jac


def _kernel_autocorrelation(kernel: FloatArray, dy: int, dx: int) -> float:
    """Autocorrelation ``sum_n k[n] * k[n + (dy, dx)]`` of a 2D kernel."""
    size = kernel.shape[0]
    ys = slice(max(0, -dy), size - max(0, dy))
    xs = slice(max(0, -dx), size - max(0, dx))
    yt = slice(max(0, dy), size - max(0, -dy))
    xt = slice(max(0, dx), size - max(0, -dx))
    return float((kernel[ys, xs] * kernel[yt, xt]).sum())


def _hvs_correlations(
    transform: np.ndarray,
    kernels: tuple[FloatArray, ...],
) -> dict[tuple[int, int], np.ndarray]:
    """Autocorrelation of the matrix HVS impulse response at each design lag.

    The matrix impulse response is ``W[n] = diag(k_c[n]) @ G`` (per-channel
    S-CIELAB spatial kernels times the linearized opponent transform), so
    its autocorrelation factors as ``Phi(Delta) = G^T diag(rho_c(Delta)) G``
    with ``rho_c`` the scalar autocorrelation of channel ``c``'s kernel.
    """
    rho = np.array(
        [[_kernel_autocorrelation(k, dy, dx) for (dy, dx) in _DESIGN_LAGS] for k in kernels]
    )
    return {
        lag: transform.T @ (rho[:, li, None] * transform) for li, lag in enumerate(_DESIGN_LAGS)
    }


def _design_dv_taps(
    transform: np.ndarray,
    kernels: tuple[FloatArray, ...],
) -> np.ndarray:
    """Solve the matrix Yule-Walker system for the optimal filter taps.

    Under DV&E's white-noise quantizer model, minimizing the perceived
    noise energy ``E||W * (u - sum_j h_j u_j)||^2`` gives the generalized
    Yule-Walker equations (their eq. 63 with eqs. 64-65 substituted):

        sum_j Phi(p_k - p_j) @ h_j = Phi(p_k),   k = 0..3

    where ``Phi(Delta) = sum_n W[n]^T W[n+Delta]``. The DC constraint
    ``sum_j h_j = I`` (their eq. 43) is imposed exactly with Lagrange
    multipliers. The taps are left-multiplied by ``Phi``, so with
    row-major vectorization the Kronecker block is ``kron(Phi, I)``, not
    the ``(X^T (x) I)`` form that ``vec(h @ X)`` would suggest.
    """
    phi = _hvs_correlations(transform, kernels)
    n = 9 * _N_TAPS
    M = np.zeros((n, n))
    rhs = np.zeros(n)
    for k_idx, pk in enumerate(_FS_POSITIONS):
        rhs[9 * k_idx : 9 * (k_idx + 1)] = phi[pk].ravel(order="C")
        for j_idx, pj in enumerate(_FS_POSITIONS):
            lag = (pk[0] - pj[0], pk[1] - pj[1])
            M[9 * k_idx : 9 * (k_idx + 1), 9 * j_idx : 9 * (j_idx + 1)] = np.kron(
                phi[lag], np.eye(3)
            )

    # Exact DC constraint via Lagrange multipliers:
    # [M S; S^T 0] [h; lambda] = [rhs; vec(I)] with S summing over taps.
    S = np.tile(np.eye(9), (_N_TAPS, 1))
    system = np.zeros((n + 9, n + 9))
    system[:n, :n] = M
    system[:n, n:] = S
    system[n:, :n] = S.T
    target = np.concatenate([rhs, np.eye(3).ravel(order="C")])
    solution = np.linalg.lstsq(system, target, rcond=None)[0]
    return solution[:n].reshape(_N_TAPS, 3, 3)


def _quantiser_gain(
    palette: FloatArray,
    train_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Linear quantizer gain K in Oklab, estimated by Monte-Carlo sampling."""
    pal_min = palette.min(axis=0) - 0.05
    pal_max = palette.max(axis=0) + 0.05
    train = rng.uniform(pal_min, pal_max, size=(train_samples, 3))
    train_q = palette[_nearest_palette_vec(train, palette)]

    x = train - train.mean(axis=0)
    x_q = train_q - train_q.mean(axis=0)
    C_x = (x.T @ x) / train_samples
    C_xq = (x_q.T @ x) / train_samples
    return C_xq @ np.linalg.inv(C_x)


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

    The design follows Damera-Venkata & Evans (2001) Sec. IV: the filter
    minimizes the perceived energy of the quantizer noise under a
    pattern-color separable HVS model (the tabulated S-CIELAB spatial
    kernels of Zhang & Wandell acting on the Poirson-Wandell opponent
    coordinates, linearized at the palette centroid), subject to the DC
    constraint that the taps sum to the identity. Under the white-noise
    quantizer model the generalized Yule-Walker system is linear and is
    solved directly: no training images or iteration are involved, and
    the design is deterministic given the palette and viewing geometry.

    Parameters
    ----------
    palette_oklab:
        Palette in Oklab, shape ``(n, 3)``.
    display_dpi:
        Pixels per inch. Together with `viewing_distance_cm` determines
        the visual-angle extent of the HVS kernels in pixel units.
    viewing_distance_cm:
        Typical viewing distance in centimeters.
    train_samples:
        Number of random colors used to estimate the quantizer gain K.
    train_size:
        Unused; retained for backward compatibility. The design is a
        direct white-noise-model solve, so no training patches are
        involved.
    seed:
        Seed for the Monte-Carlo estimation of K (reproducibility).

    Returns
    -------
    :class:`DVFilters` with the designed taps and quantizer gain matrix.

    Raises
    ------
    ValueError
        If `palette_oklab` has fewer than 2 colors.

    """
    rng = np.random.default_rng(seed)
    palette = np.asarray(palette_oklab, dtype=np.float64)
    if len(palette) < 2:
        raise ValueError(f"dv_filters requires at least 2 palette colors, got {len(palette)}")

    # The linearized opponent transform at the palette centroid. For a
    # degenerate (near-black) centroid the Oklab->XYZ cube makes the
    # Jacobian ill-conditioned; fall back to scalar Floyd-Steinberg taps,
    # which are DC-conserving and unconditionally stable.
    transform = _opponent_jacobian(palette.mean(axis=0))
    if not np.all(np.isfinite(transform)) or np.linalg.cond(transform) > 1e8:
        weights = np.array([np.eye(3) * w for _, _, w in _FS_WEIGHTS_FALLBACK])
    else:
        kernels = _scielab_kernels(display_dpi, viewing_distance_cm)
        weights = _design_dv_taps(transform, kernels)

    K = _quantiser_gain(palette, train_samples, rng)

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
    clamp_margin: float | None = _CLAMP_MARGIN,
) -> tuple[FloatArray, np.ndarray]:
    """Quantize ``image`` to ``palette`` using matrix-valued DV error diffusion.

    Diffuses in Oklab, the space the palette and the quality metrics live
    in, with matrix-valued filter taps. Diffusing in a nonlinearly related
    working space (e.g. the opponent space itself) conserves that space's
    mean instead of Oklab's and shows up as low-frequency tone error in the
    metrics; the opponent structure of the HVS model is instead folded into
    the taps at design time via the linearized transform.

    ``shaping``, if given, must be a single ``(3, 3)`` global shaping
    matrix; it is applied to the Oklab error before distribution. Per-cell
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
    clamp_margin:
        Margin (in Oklab units) outside the palette's per-channel bounding
        box within which the accumulation buffer may drift before being
        clamped. HVS-optimal filters are aggressive enough to push the real
        quantizer loop into overload (the buffer drifts far outside the
        palette range, the quantizer's effective gain exceeds one, and the
        loop oscillates), which the linear design model cannot see; the
        clamp is the standard practical stabilizer. ``None`` disables it.

    Returns
    -------
    ``(dithered, indices)``: the quantized image in Oklab and the
    per-pixel palette index.

    Raises
    ------
    ValueError
        If `shaping` is not a single ``(3, 3)`` matrix; per-cell
        ``(n, 3, 3)`` shaping matrices are not supported.

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

    weights = filters.weights

    height, width = image_oklab.shape[:2]
    buf = image_oklab.copy()
    indices = np.zeros((height, width), dtype=np.int64)

    if clamp_margin is not None:
        buf_lo = palette_oklab.min(axis=0) - clamp_margin
        buf_hi = palette_oklab.max(axis=0) + clamp_margin

    for y, xs, sign in _scanlines(height, width, serpentine):
        for x in xs:
            pixel = buf[y, x]
            index = nearest_palette_index(pixel, palette_oklab)
            indices[y, x] = index
            error = pixel - palette_oklab[index]

            if shaping is not None:
                error = shaping @ error

            for k, (dy, dx) in enumerate(_FS_POSITIONS):
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny, nx] += weights[k] @ error
                    if clamp_margin is not None:
                        np.clip(buf[ny, nx], buf_lo, buf_hi, out=buf[ny, nx])

    return palette_oklab[indices], indices
