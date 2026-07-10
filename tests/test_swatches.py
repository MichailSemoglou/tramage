"""Tests for swatch-file palette import (.ase, .gpl, hex lists)."""

import struct

import numpy as np
import pytest

from tramage.colour import oklab_to_srgb
from tramage.swatches import load_ase, load_gpl, load_hex_list, load_swatch


def test_load_hex_list_hash_prefixed(tmp_path):
    path = tmp_path / "palette.txt"
    path.write_text("#FF5733\n#33FF57\n#3357FF\n")
    palette, names = load_hex_list(path)
    assert palette.shape == (3, 3)
    assert names == ["#FF5733", "#33FF57", "#3357FF"]
    rgb = np.clip(oklab_to_srgb(palette), 0, 1)
    assert np.allclose(rgb[0], [0xFF / 255, 0x57 / 255, 0x33 / 255], atol=1e-3)


def test_load_hex_list_comma_separated_one_line(tmp_path):
    path = tmp_path / "palette.txt"
    path.write_text("#FF5733, #33FF57, #3357FF")
    palette, names = load_hex_list(path)
    assert palette.shape == (3, 3)
    assert len(names) == 3


def test_load_hex_list_bare_hex_fallback(tmp_path):
    path = tmp_path / "palette.txt"
    path.write_text("FF5733\n33FF57\n3357FF\n")
    palette, names = load_hex_list(path)
    assert palette.shape == (3, 3)
    assert names == ["#FF5733", "#33FF57", "#3357FF"]


def test_load_hex_list_shorthand_three_digit(tmp_path):
    path = tmp_path / "palette.txt"
    path.write_text("#f53\n")
    palette, names = load_hex_list(path)
    assert palette.shape == (1, 3)
    assert names == ["#FF5533"]


def test_load_hex_list_raises_when_nothing_found(tmp_path):
    path = tmp_path / "palette.txt"
    path.write_text("this file has no colours in it at all")
    with pytest.raises(ValueError):
        load_hex_list(path)


def test_load_gpl_basic(tmp_path):
    path = tmp_path / "palette.gpl"
    path.write_text(
        "GIMP Palette\n"
        "Name: Test\n"
        "Columns: 3\n"
        "#\n"
        "255   0   0\tRed\n"
        "  0 255   0\tGreen\n"
        "  0   0 255\tBlue\n"
    )
    palette, names = load_gpl(path)
    assert palette.shape == (3, 3)
    assert names == ["Red", "Green", "Blue"]


def test_load_gpl_without_names_falls_back_to_hex(tmp_path):
    path = tmp_path / "palette.gpl"
    path.write_text("GIMP Palette\n255 0 0\n")
    _, names = load_gpl(path)
    assert names == ["#FF0000"]


def test_load_gpl_rejects_non_gpl_file(tmp_path):
    path = tmp_path / "palette.gpl"
    path.write_text("not a gimp palette\n255 0 0\n")
    with pytest.raises(ValueError):
        load_gpl(path)


def test_load_gpl_clips_out_of_range_values(tmp_path):
    """A malformed or hand-edited file with an out-of-range value should be
    clipped rather than silently producing a colour outside the sRGB gamut."""
    path = tmp_path / "palette.gpl"
    path.write_text("GIMP Palette\n300 0 0\tOut of range\n")
    palette, _ = load_gpl(path)
    rgb = np.clip(oklab_to_srgb(palette), 0, 1)
    assert np.allclose(rgb[0], [1.0, 0.0, 0.0], atol=1e-3)


def _pack_ase_entry(name: str, model: bytes, values: tuple) -> bytes:
    name_utf16 = (name + "\x00").encode("utf-16-be")
    name_len = len(name_utf16) // 2
    data = struct.pack(">H", name_len) + name_utf16 + model
    data += struct.pack(f">{len(values)}f", *values)
    data += struct.pack(">H", 0)  # colour type: Global
    return struct.pack(">HI", 0x0001, len(data)) + data


def _build_ase_bytes(entries: list[bytes]) -> bytes:
    header = b"ASEF" + struct.pack(">HHI", 1, 0, len(entries))
    return header + b"".join(entries)


def test_load_ase_rgb_and_cmyk_and_gray(tmp_path):
    entries = [
        _pack_ase_entry("Red", b"RGB ", (1.0, 0.0, 0.0)),
        _pack_ase_entry("Half Gray", b"Gray", (0.5,)),
        _pack_ase_entry("Black CMYK", b"CMYK", (0.0, 0.0, 0.0, 1.0)),
    ]
    path = tmp_path / "palette.ase"
    path.write_bytes(_build_ase_bytes(entries))

    palette, names = load_ase(path)
    assert palette.shape == (3, 3)
    assert names == ["Red", "Half Gray", "Black CMYK"]

    rgb = np.clip(oklab_to_srgb(palette), 0, 1)
    assert np.allclose(rgb[0], [1.0, 0.0, 0.0], atol=1e-3)
    assert np.allclose(rgb[1], [0.5, 0.5, 0.5], atol=1e-3)
    assert np.allclose(rgb[2], [0.0, 0.0, 0.0], atol=1e-3)


def test_load_ase_skips_lab_with_warning(tmp_path):
    entries = [
        _pack_ase_entry("Red", b"RGB ", (1.0, 0.0, 0.0)),
        _pack_ase_entry("Some Lab Colour", b"LAB ", (50.0, 10.0, -10.0)),
    ]
    path = tmp_path / "palette.ase"
    path.write_bytes(_build_ase_bytes(entries))

    with pytest.warns(UserWarning, match="LAB"):
        palette, names = load_ase(path)
    assert palette.shape == (1, 3)
    assert names == ["Red"]


def test_load_ase_rejects_bad_signature(tmp_path):
    path = tmp_path / "palette.ase"
    path.write_bytes(b"NOTASE" + b"\x00" * 10)
    with pytest.raises(ValueError):
        load_ase(path)


def test_load_ase_rejects_truncated_file(tmp_path):
    """Header claims 2 blocks but the file only contains 1.

    Previously this leaked a raw, cryptic `struct.error` instead of the
    module's own clear `ValueError`.
    """
    entry = _pack_ase_entry("Red", b"RGB ", (1.0, 0.0, 0.0))
    header = b"ASEF" + struct.pack(">HHI", 1, 0, 2)
    path = tmp_path / "palette.ase"
    path.write_bytes(header + entry)
    with pytest.raises(ValueError, match="truncated"):
        load_ase(path)


def test_load_ase_rejects_oversized_block_length(tmp_path):
    """A block declares a length longer than the bytes that actually follow.

    Previously this was silently accepted: slicing truncated the block
    without error, and the truncated bytes were misparsed as a garbled
    fake swatch instead of raising.
    """
    bad_entry = struct.pack(">HI", 0x0001, 9999) + b"short data"
    header = b"ASEF" + struct.pack(">HHI", 1, 0, 1)
    path = tmp_path / "palette.ase"
    path.write_bytes(header + bad_entry)
    with pytest.raises(ValueError, match="truncated"):
        load_ase(path)


def test_load_ase_rejects_file_shorter_than_header(tmp_path):
    path = tmp_path / "palette.ase"
    path.write_bytes(b"ASEF\x00\x01")
    with pytest.raises(ValueError, match="truncated"):
        load_ase(path)


def test_load_swatch_dispatches_by_extension(tmp_path):
    ase_path = tmp_path / "palette.ase"
    ase_path.write_bytes(_build_ase_bytes([_pack_ase_entry("Red", b"RGB ", (1.0, 0.0, 0.0))]))
    palette, names = load_swatch(ase_path)
    assert names == ["Red"]

    gpl_path = tmp_path / "palette.gpl"
    gpl_path.write_text("GIMP Palette\n255 0 0\tRed\n")
    palette, names = load_swatch(gpl_path)
    assert names == ["Red"]

    hex_path = tmp_path / "palette.txt"
    hex_path.write_text("#FF0000\n")
    palette, names = load_swatch(hex_path)
    assert names == ["#FF0000"]
