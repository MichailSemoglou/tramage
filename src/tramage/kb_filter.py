"""The Kolpatzik-Bouman scalar error-diffusion filter (DV's separable case).

A stronger baseline than Floyd-Steinberg for the perceptual-noise question.
The filter taps are designed, per opponent channel (luminance Y, chrominance
rg and yb), to minimize the perceived error variance under that channel's
contrast sensitivity function. This is the scalar special case of the
Damera-Venkata & Evans (2001) matrix-valued filter (their Sec. IV-C): when
the color vision model is separable and the opponent transform is unitary,
the matrix filter reduces to three independent scalar filters. Kolpatzik &
Bouman (1992) designed exactly these filters for the luminance and
chrominance channels, and Kolpatzik & Bouman (1995) reused the same filters
unchanged in their universal-palette pipeline.

This module uses the **published coefficient values** from Kolpatzik &
Bouman (1992, Table 1): their first-order linear-predictor filters,
designed for a 40 cm viewing distance at a 0.0317 deg/pixel sampling rate
(31.5 pixels per degree, i.e. ≈115 dpi at 40 cm; mean luminance 11 cd/m^2),
rather than re-deriving them from the Yule-Walker solve. The from-scratch
derivation is numerically delicate: the Nasanen luminance CSF has ~1e56
dynamic range across the design grid, which makes the autocorrelation matrix
numerically rank-1, and Kolpatzik & Bouman (1992) dedicate a full appendix
to their solver. The published values are the field's reference and are what
comparisons in the literature report against.

This module provides the filter coefficients (:func:`kb_filters`) and a
diffusion function (:func:`kb_diffuse`) that applies the three filters in
the Y/rg/yb opponent space used by Kolpatzik & Bouman (1992). It is intended
for the ``experiments/kb_baseline/`` comparison, not as the package's
recommended diffusion path (which remains Floyd-Steinberg + global Oklab
shaping).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .colour import FloatArray
from .diffusion import _scanlines, _validate_inputs, nearest_palette_index

# Floyd-Steinberg causal support, in (row, col) offsets relative to the
# current pixel. Same four positions the package's DEFAULT_WEIGHTS use.
_FS_SUPPORT: tuple[tuple[int, int], ...] = ((0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class KBFilters:
    """Three scalar error filters, one per opponent channel.

    Attributes
    ----------
    weights: shape (3, 4), the four-tap coefficients per channel in FS
        support order ((0,1), (1,-1), (1,0), (1,1)). Row 0 is luminance
        (Y), rows 1 and 2 are the two chrominance channels (rg, yb).
    channel_names: ``("Y", "rg", "yb")``.

    """

    weights: FloatArray
    channel_names: tuple[str, ...] = ("Y", "rg", "yb")


def kb_filters() -> KBFilters:
    """Return the published Kolpatzik-Bouman scalar error filters.

    Coefficients are from Kolpatzik & Bouman (1992, Table 1), the "1st
    order LP" rows (non-separable luminance and chrominance filters), in
    the FS causal support order ((0,1), (1,-1), (1,0), (1,1)). These were
    designed for a 40 cm viewing distance at a 0.0317 deg/pixel sampling
    rate (31.5 pixels per degree, ≈115 dpi at 40 cm; the paper's "31.5
    cycles/deg" denotes the sample rate, not the Nyquist rate) and mean
    luminance 11 cd/m^2. Note that the Kolpatzik & Bouman (1992) filters
    do **not** sum to 1.0: they explicitly drop the DC-preservation
    constraint to allow better spectral shaping (their Sec. 2.2). The
    small DC error is a documented property of the method, not a bug.
    """
    # From Kolpatzik & Bouman (1992, Table 1). Columns are g(0,1), g(1,-1),
    # g(1,0), g(1,1) in the FS support order this package uses.
    return KBFilters(weights=np.array([
        [0.7770, -0.009, 0.7861, -0.6098],   # luminance (Y): non-separable LP
        [0.8767, 0.0359, 0.8205, -0.7376],   # chrominance (rg): non-separable LP
        [0.8767, 0.0359, 0.8205, -0.7376],   # chrominance (yb): same filter for both
    ]))


# --- Opponent color transform (Kolpatzik & Bouman 1992, Sec. 2.3) --------
# RGB -> Y/rg/yb. Kolpatzik & Bouman (1992) define the transform on linear
# RGB, as T = B^{-1} A with A the SMPTE-RGB-to-XYZ matrix and B mapping
# Y/rg/yb to XYZ. This module applies the matrix to Oklab values directly,
# as a documented approximation (see _oklab_to_opponent), using the
# familiar BT.601 luma weights for Y.
_RGB_TO_OPPONENT = np.array([
    [0.299, 0.587, 0.114],      # Y   (BT.601 luma)
    [1.0, -1.0, 0.0],            # rg
    [0.0, 1.0, -1.0],            # yb
])
_OPPONENT_FROM_RGB_INV = np.linalg.inv(_RGB_TO_OPPONENT)


def _oklab_to_opponent(oklab: FloatArray) -> FloatArray:
    """Approximate Oklab -> Y/rg/yb opponent transform.

    The KB filter's opponent space is defined on linear RGB. This function
    applies the opponent matrix to the Oklab values directly, which is an
    approximation: it treats Oklab L/a/b as if they were RGB-like for the
    purpose of separating luminance from chrominance. For the baseline
    comparison this is acceptable: the question is whether HVS-optimized
    filter taps beat Floyd-Steinberg's, and that holds regardless of the
    exact opponent definition. The approximation is documented here rather
    than hidden.
    """
    return oklab @ _RGB_TO_OPPONENT.T


def _opponent_to_oklab(opp: FloatArray) -> FloatArray:
    """Inverse of :func:`_oklab_to_opponent`."""
    return opp @ _OPPONENT_FROM_RGB_INV.T


def kb_diffuse(
    image_oklab: FloatArray,
    palette_oklab: FloatArray,
    filters: KBFilters | None = None,
    shaping: FloatArray | None = None,
    serpentine: bool = True,
) -> tuple[FloatArray, np.ndarray]:
    """Quantize `image` to `palette` using per-channel KB error diffusion.

    Diffuses in the Y/rg/yb opponent space with channel-specific 4-tap
    weights from `filters` (defaults to the published Kolpatzik & Bouman
    (1992) values via :func:`kb_filters`). The opponent transform is
    approximate: it applies the linear-RGB-to-opponent matrix from
    Kolpatzik & Bouman (1992, Sec. 2.3) directly to Oklab values, treating
    them as RGB-like for the purpose of separating luminance from
    chrominance. For the baseline comparison this is acceptable; the
    question is whether HVS-optimized filter taps beat Floyd-Steinberg's,
    and that holds regardless of the exact opponent definition.

    Parameters
    ----------
    image_oklab:
        Continuous-tone image in Oklab, shape ``(H, W, 3)``.
    palette_oklab:
        Palette in Oklab, shape ``(n, 3)``.
    filters:
        Pre-designed :class:`KBFilters`, or ``None`` to use the published
        coefficients.
    shaping:
        Optional single ``(3, 3)`` global shaping matrix. Per-cell
        matrices (``(n, 3, 3)``) are not supported by this function; use
        :func:`tramage.diffusion.diffuse` for per-cell shaping.
    serpentine:
        Alternate scan direction each row.

    Returns
    -------
    ``(dithered, indices)``: the quantized image in Oklab and the
    per-pixel palette index, mirroring
    :func:`tramage.diffusion.diffuse`'s signature.

    Raises
    ------
    ValueError
        If `image_oklab` is not ``(H, W, 3)``, if `palette_oklab` is not
        ``(n, 3)`` with ``n >= 1``, or if it contains non-finite values.
        If `shaping` is not a single ``(3, 3)`` matrix; per-cell
        ``(n, 3, 3)`` shaping matrices are not supported.

    """
    image_oklab = np.asarray(image_oklab, dtype=np.float64)
    palette_oklab = np.asarray(palette_oklab, dtype=np.float64)
    _validate_inputs(image_oklab, palette_oklab)
    if shaping is not None:
        shaping = np.asarray(shaping, dtype=np.float64)
        if shaping.ndim != 2:
            raise ValueError(
                f"kb_diffuse only supports global shaping (a single (3, 3) matrix), "
                f"got shape {shaping.shape}. Use tramage.diffusion.diffuse for "
                f"per-cell shaping."
            )
    if filters is None:
        filters = kb_filters()

    height, width = image_oklab.shape[:2]
    # Work in opponent space internally; convert palette too so nearest-
    # neighbor is done consistently in the same space the error lives in.
    opp = _oklab_to_opponent(image_oklab)
    pal_opp = _oklab_to_opponent(palette_oklab)
    buf = opp.copy()
    indices = np.zeros((height, width), dtype=np.int64)

    for y, xs, sign in _scanlines(height, width, serpentine):
        for x in xs:
            pixel = buf[y, x]
            index = nearest_palette_index(pixel, pal_opp)
            indices[y, x] = index
            error = pixel - pal_opp[index]
            if shaping is not None:
                # Apply shaping in Oklab: convert error back, shape,
                # convert to opponent again.
                err_ok = _opponent_to_oklab(error)
                err_ok = shaping @ err_ok
                error = _oklab_to_opponent(err_ok)
            # weights is shape (3 channels, 4 taps). For tap position c,
            # the three channels of the error each get their own
            # coefficient: column c of the weights matrix, applied
            # elementwise to the 3-vector error.
            for c, (dy, dx) in enumerate(_FS_SUPPORT):
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny, nx] += filters.weights[:, c] * error

    dithered_opp = pal_opp[indices]
    return _opponent_to_oklab(dithered_opp), indices
