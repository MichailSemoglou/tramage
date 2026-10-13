"""Serpentine error diffusion with pluggable error shaping.

This is ordinary Floyd-Steinberg error diffusion in its serpentine
(boustrophedon) form: alternate rows scan left-to-right and right-to-left,
which avoids the tendency of raster-only scanning to drag error in one
horizontal direction. The only departure from the textbook algorithm is
that the quantization error is passed through a shaping step before it is
distributed to neighboring pixels. Pass ``shaping=None`` and this is
exactly vector Floyd-Steinberg; pass a matrix or per-color matrices from
:mod:`tramage.shaping` and the error is reshaped along the palette's own
structure first.
"""

from __future__ import annotations

import numpy as np

from .colour import FloatArray

__all__ = ["DEFAULT_WEIGHTS", "diffuse"]

# Floyd-Steinberg kernel as (row offset, column offset, weight). Column
# offset is mirrored by the scan direction on odd rows.
DEFAULT_WEIGHTS: tuple[tuple[int, int, float], ...] = (
    (0, 1, 7 / 16),
    (1, -1, 3 / 16),
    (1, 0, 5 / 16),
    (1, 1, 1 / 16),
)


def nearest_palette_index(pixel: FloatArray, palette: FloatArray) -> int:
    """Index of the palette color closest to `pixel` in Euclidean distance."""
    return int(np.argmin(((palette - pixel) ** 2).sum(axis=-1)))


def _shape_error(e: FloatArray, index: int, shaping: FloatArray | None) -> FloatArray:
    if shaping is None:
        return e
    if shaping.ndim == 2:
        return shaping @ e
    if shaping.ndim == 3:
        return shaping[index] @ e
    raise ValueError(
        f"shaping must be a (3, 3) or (n, 3, 3) array, got shape {shaping.shape}"
    )


def _scanlines(
    height: int,
    width: int,
    serpentine: bool = True,
):
    """Yield ``(y, xs, sign)`` for each row of a serpentine or raster scan.

    ``xs`` is ``range(width)`` on left-to-right rows and
    ``range(width-1, -1, -1)`` on right-to-left rows. ``sign`` is +1 for
    left-to-right and -1 for right-to-left, for mirroring column offsets in
    error-diffusion kernels.

    This is a generator: rows are produced one at a time with O(width)
    total memory, not O(height × width).
    """
    for y in range(height):
        left_to_right = not serpentine or y % 2 == 0
        xs = range(width) if left_to_right else range(width - 1, -1, -1)
        sign = 1 if left_to_right else -1
        yield y, xs, sign


def diffuse(
    image: FloatArray,
    palette: FloatArray,
    shaping: FloatArray | None = None,
    weights: tuple[tuple[int, int, float], ...] = DEFAULT_WEIGHTS,
    serpentine: bool = True,
) -> tuple[FloatArray, np.ndarray]:
    """Quantize `image` to `palette` by serpentine error diffusion.

    Parameters
    ----------
    image:
        Continuous-tone image in the same color space as `palette` (Oklab,
        in this package), shape ``(H, W, 3)``.
    palette:
        Candidate colors, shape ``(n, 3)``.
    shaping:
        ``None`` for unshaped vector Floyd-Steinberg. A single ``(3, 3)``
        matrix to apply the same shaping everywhere (see
        :func:`tramage.shaping.global_shaping_matrix`). An ``(n, 3, 3)``
        array to apply the matrix for the chosen palette color at each
        pixel (see :func:`tramage.shaping.per_cell_shaping_matrices`).
    weights:
        Error-diffusion kernel as ``(row offset, column offset, weight)``
        triples. Defaults to Floyd-Steinberg's 7/3/5/1 over 16.
    serpentine:
        Alternate scan direction each row. Set `False` for plain
        left-to-right raster order.

    Returns
    -------
    A tuple ``(dithered, indices)``: the quantized image made of palette
    colors, and the palette index chosen at each pixel.

    """
    image = np.asarray(image, dtype=np.float64)
    palette = np.asarray(palette, dtype=np.float64)
    if shaping is not None:
        shaping = np.asarray(shaping, dtype=np.float64)

    height, width = image.shape[:2]
    buf = image.copy()
    indices = np.zeros((height, width), dtype=np.int64)

    for y, xs, sign in _scanlines(height, width, serpentine):
        for x in xs:
            pixel = buf[y, x]
            index = nearest_palette_index(pixel, palette)
            indices[y, x] = index
            error = _shape_error(pixel - palette[index], index, shaping)
            for dy, dx, weight in weights:
                ny, nx = y + dy, x + sign * dx
                if 0 <= ny < height and 0 <= nx < width:
                    buf[ny, nx] += weight * error

    return palette[indices], indices
