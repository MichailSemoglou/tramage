"""Tests for palette extraction and eigenspectrum reporting."""

import numpy as np
import pytest

from tramage.palette import describe_palette, eigenspectrum_report, extract_palette


def test_extract_palette_shape_and_shares_sum_to_one():
    rng = np.random.default_rng(0)
    image = rng.uniform(size=(40, 40, 3))
    palette, shares = extract_palette(image, n_colours=6, seed=0)
    assert palette.shape == (6, 3)
    assert shares.shape == (6,)
    assert np.isclose(shares.sum(), 1.0)


def test_extract_palette_shares_descending():
    rng = np.random.default_rng(1)
    image = rng.uniform(size=(40, 40, 3))
    _, shares = extract_palette(image, n_colours=5, seed=0)
    assert np.all(np.diff(shares) <= 0)


def test_extract_palette_missing_raise_on_degenerate_image():
    """Requesting more colours than the image actually has should be
    reportable as an error when the caller asks for it, rather than only
    ever silently returning a duplicate, 0%-share colour."""
    flat_image = np.zeros((10, 10, 3))
    with pytest.raises(Exception):
        extract_palette(flat_image, n_colours=5, seed=0, missing="raise")


def test_eigenspectrum_report_str_contains_eigenvalues():
    Ls, As = [0.40, 0.55, 0.70], [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, 0.02] for L in Ls for a in As])
    report = eigenspectrum_report(palette)
    assert report.eigenvalues.shape == (3,)
    assert "eigenvalues" in str(report)


def test_describe_palette_isotropic_says_little_to_do():
    rng = np.random.default_rng(5)
    palette = rng.normal(loc=[0.5, 0.0, 0.0], scale=0.2, size=(30, 3))
    description = describe_palette(palette, eps=1e-8)
    assert "real extent in all three directions" in description
    assert "little to do" in description


def test_describe_palette_rank_deficient_says_real_extent_and_flat():
    """The standard nine-colour rank-deficient palette (b held constant)
    has real extent in two of three directions and is flat in the third."""
    Ls, As = [0.40, 0.55, 0.70], [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, 0.02] for L in Ls for a in As])
    description = describe_palette(palette, eps=0.02 ** 2)
    assert "real extent in 2 of 3 directions" in description
    assert "1 close to flat" in description
    assert "likely to matter" in description


def test_describe_palette_fully_degenerate_palette():
    """A palette with zero spread in every direction (two identical
    colours) should be described as such, not as a division-by-zero crash."""
    palette = np.array([[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]])
    description = describe_palette(palette, eps=0.02 ** 2)
    assert "almost no measurable spread" in description


def test_describe_palette_does_not_recommend_an_eps_value():
    """The function must describe geometry relative to the given eps, not
    tell the caller what eps should be; that heuristic is not yet settled."""
    rng = np.random.default_rng(6)
    palette = rng.normal(size=(10, 3))
    description = describe_palette(palette)
    assert "you should" not in description.lower()
    assert "we recommend" not in description.lower()
    assert "try eps" not in description.lower()


def test_describe_palette_reports_uniform_locality_for_isotropic_palette():
    rng = np.random.default_rng(7)
    palette = rng.normal(loc=[0.5, 0.0, 0.0], scale=0.2, size=(30, 3))
    description = describe_palette(palette, eps=1e-8)
    assert "fairly uniform" in description


def test_eigenspectrum_report_per_cell_false_skips_per_cell_cost():
    """per_cell=False should return global eigenvalues without computing
    per-cell gains (which are O(n²)). The per-cell fields become zero."""
    Ls, As = [0.40, 0.55, 0.70], [-0.08, 0.00, 0.08]
    palette = np.array([[L, a, 0.02] for L in Ls for a in As])
    report = eigenspectrum_report(palette, per_cell=False)
    assert report.eigenvalues.shape == (3,)
    assert report.gains.shape == (3,)
    assert np.all(report.per_cell_gain_min == 0.0)
    assert np.all(report.per_cell_gain_max == 0.0)
