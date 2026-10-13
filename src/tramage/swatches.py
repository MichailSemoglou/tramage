"""Import palettes from common design-tool swatch files.

Studios and designers usually already have their actual palette as a file,
not a photograph to re-derive it from: an Adobe Swatch Exchange (``.ase``)
file, a GIMP palette (``.gpl``), or a plain list of hex codes. This module
reads the formats designers overwhelmingly use and converts them straight
to Oklab, so a real brand palette can be used with :mod:`tramage.shaping`
and :mod:`tramage.diffusion` without going through an image at all.

Scope is deliberately narrow rather than attempting universal swatch
support. Adobe Swatch Exchange is a semi-documented binary format that has
drifted slightly between Adobe application versions; this parser reads
RGB, Gray and CMYK color entries, which are the overwhelming majority
found in real files, and skips LAB entries with a warning rather than
silently misconverting them: ASE's Lab encoding would need a dedicated
conversion path that is not implemented here. CMYK entries are converted
to sRGB with a naive, uncalibrated formula and should be treated as an
approximation, not a color-managed conversion.
"""

from __future__ import annotations

import re
import struct
import warnings
from pathlib import Path

import numpy as np

from .colour import FloatArray, srgb_to_oklab

_HEX_WITH_HASH_RE = re.compile(r"#([0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})\b")
_BARE_HEX_RE = re.compile(r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{3})$")


def _hex_to_srgb(token: str) -> tuple[float, float, float]:
    if len(token) == 3:
        token = "".join(ch * 2 for ch in token)
    r, g, b = (int(token[i:i + 2], 16) for i in (0, 2, 4))
    return r / 255.0, g / 255.0, b / 255.0


def _clamp_channel(v: float) -> int:
    return max(0, min(255, round(v * 255)))


def _srgb_to_hex_name(r: float, g: float, b: float) -> str:
    return f"#{_clamp_channel(r):02X}{_clamp_channel(g):02X}{_clamp_channel(b):02X}"


def load_hex_list(path: str | Path) -> tuple[FloatArray, list[str]]:
    """Load a plain text file of hex color codes, returning (palette, names).

    Accepts one code per line, several separated by commas or whitespace on
    one line, with or without a leading ``#``, and 3- or 6-digit forms.
    Anything that is not a recognizable hex token (headers, comments,
    surrounding text from a Coolors.co-style export) is ignored rather than
    treated as an error. If no ``#``-prefixed tokens are found at all, each
    line is tried as a bare hex code on its own. Names are the hex code
    itself, since plain hex lists rarely carry a separate color name.

    Raises
    ------
    ValueError
        If the file contains no recognizable hex color codes.

    """
    text = Path(path).read_text()
    tokens = list(_HEX_WITH_HASH_RE.findall(text))
    if not tokens:
        for line in text.splitlines():
            stripped = line.strip()
            if _BARE_HEX_RE.match(stripped):
                tokens.append(stripped)
    if not tokens:
        raise ValueError(f"no hex color codes found in {path!s}")

    srgb = np.array([_hex_to_srgb(t) for t in tokens], dtype=np.float64)
    names = [
        f"#{t.upper()}" if len(t) == 6 else f"#{''.join(c * 2 for c in t.upper())}"
        for t in tokens
    ]
    return srgb_to_oklab(srgb), names


def load_gpl(path: str | Path) -> tuple[FloatArray, list[str]]:
    """Load a GIMP palette (``.gpl``) file, returning (palette, names).

    Follows the standard GPL text format: a ``GIMP Palette`` header line,
    optional ``Name:`` and ``Columns:`` lines, ``#``-prefixed comment
    lines, and one color per remaining line as three whitespace-separated
    0-255 integers, optionally followed by a name for the rest of the line.

    Raises
    ------
    ValueError
        If the file does not start with a ``GIMP Palette`` header, or
        contains no color entries.

    """
    lines = Path(path).read_text().splitlines()
    if not lines or not lines[0].strip().upper().startswith("GIMP PALETTE"):
        raise ValueError(f"{path!s} does not look like a GIMP palette (.gpl) file")

    srgb: list[tuple[float, float, float]] = []
    names: list[str] = []
    for line in lines[1:]:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.lower().startswith(("name:", "columns:")):
            continue
        parts = stripped.split(maxsplit=3)
        if len(parts) < 3:
            continue
        try:
            r, g, b = (int(parts[i]) for i in range(3))
        except ValueError:
            continue
        srgb.append((r / 255.0, g / 255.0, b / 255.0))
        names.append(
            parts[3] if len(parts) > 3 else _srgb_to_hex_name(r / 255.0, g / 255.0, b / 255.0)
        )

    if not srgb:
        raise ValueError(f"no color entries found in {path!s}")
    # Well-formed GPL files always have 0-255 integers, but clip defensively
    # in case of a malformed or hand-edited file with an out-of-range value.
    clipped = np.clip(np.array(srgb, dtype=np.float64), 0.0, 1.0)
    return srgb_to_oklab(clipped), names


def load_ase(path: str | Path) -> tuple[FloatArray, list[str]]:
    """Load an Adobe Swatch Exchange (``.ase``) file, returning (palette, names).

    Reads RGB, Gray and CMYK color entries; group start and end markers
    are read and discarded, so the palette is returned flat, without group
    structure. LAB entries are skipped with a warning, because ASE's Lab
    encoding would need a dedicated conversion path that is not implemented
    here; CMYK entries are converted with a naive, uncalibrated formula
    and should be treated as approximate.

    Raises
    ------
    ValueError
        If the file is not an ASE file (bad signature), is truncated or
        corrupt, or contains no usable color entries.

    """
    data = Path(path).read_bytes()
    if data[:4] != b"ASEF":
        raise ValueError(f"{path!s} does not look like an ASE file (bad signature)")
    if len(data) < 12:
        raise ValueError(f"truncated ASE file (no header): {path!s}")

    offset = 4
    _major, _minor, n_blocks = struct.unpack_from(">HHI", data, offset)
    offset += 8

    srgb: list[tuple[float, float, float]] = []
    names: list[str] = []
    for _ in range(n_blocks):
        if offset + 6 > len(data):
            raise ValueError(f"truncated ASE file (block header cut off): {path!s}")
        block_type, block_length = struct.unpack_from(">HI", data, offset)
        offset += 6
        if offset + block_length > len(data):
            raise ValueError(f"truncated ASE file (block body cut off): {path!s}")
        block = data[offset:offset + block_length]
        offset += block_length

        if block_type != 0x0001:  # group start (0xC001) or end (0xC002)
            continue

        try:
            pos = 0
            (name_len,) = struct.unpack_from(">H", block, pos)
            pos += 2
            name = (
                block[pos:pos + name_len * 2]
                .decode("utf-16-be", errors="replace")
                .rstrip("\x00")
            )
            pos += name_len * 2

            model = block[pos:pos + 4].decode("ascii", errors="replace")
            pos += 4

            if model == "RGB ":
                r, g, b = struct.unpack_from(">3f", block, pos)
            elif model == "Gray":
                (gray,) = struct.unpack_from(">f", block, pos)
                r = g = b = gray
            elif model == "CMYK":
                c, m, y, k = struct.unpack_from(">4f", block, pos)
                r, g, b = (1 - c) * (1 - k), (1 - m) * (1 - k), (1 - y) * (1 - k)
            elif model == "LAB ":
                warnings.warn(
                    f"skipping LAB swatch {name!r} in {path!s}: "
                    "ASE's Lab encoding requires a dedicated conversion path "
                    "that tramage does not implement",
                    stacklevel=2,
                )
                continue
            else:
                warnings.warn(
                    f"skipping swatch {name!r} with unrecognized color model {model!r}",
                    stacklevel=2,
                )
                continue
        except struct.error as exc:
            raise ValueError(f"corrupt ASE color entry in {path!s}") from exc

        srgb.append((r, g, b))
        names.append(name or _srgb_to_hex_name(r, g, b))

    if not srgb:
        raise ValueError(f"no usable color entries found in {path!s}")
    clipped = np.clip(np.array(srgb, dtype=np.float64), 0.0, 1.0)
    return srgb_to_oklab(clipped), names


def load_swatch(path: str | Path) -> tuple[FloatArray, list[str]]:
    """Load a palette from a swatch file, dispatching on its file extension.

    Supports Adobe Swatch Exchange (``.ase``) and GIMP palette (``.gpl``)
    files by extension; anything else is tried as a plain text hex color
    list.

    Raises
    ------
    ValueError
        Propagated from the delegated loader when the file is malformed.

    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".ase":
        return load_ase(path)
    if suffix == ".gpl":
        return load_gpl(path)
    return load_hex_list(path)
