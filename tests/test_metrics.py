"""Tests for fidelity and noise metrics."""

import numpy as np

from tramage.metrics import (
    blurred_tone_error,
    lowfreq_energy_fraction,
    noise_std,
    scielab_de,
    spectral_anisotropy,
    systematic_bias,
)


def test_identical_images_have_zero_bias_and_noise():
    rng = np.random.default_rng(8)
    image = rng.uniform(size=(32, 32, 3))
    assert np.allclose(systematic_bias(image, image), 0.0)
    assert np.allclose(noise_std(image, image), 0.0)
    assert np.isclose(blurred_tone_error(image, image), 0.0)


def test_systematic_bias_detects_constant_offset():
    rng = np.random.default_rng(9)
    original = rng.uniform(size=(20, 20, 3))
    offset = np.array([0.1, -0.05, 0.02])
    dithered = original + offset
    bias = systematic_bias(dithered, original)
    assert np.allclose(bias, offset, atol=1e-10)


def test_noise_std_scales_with_error_magnitude():
    rng = np.random.default_rng(10)
    original = rng.uniform(size=(64, 64, 3))
    small_noise = original + rng.normal(scale=0.01, size=original.shape)
    large_noise = original + rng.normal(scale=0.10, size=original.shape)
    assert np.all(noise_std(large_noise, original) > noise_std(small_noise, original))


def test_lowfreq_energy_fraction_is_between_zero_and_one():
    rng = np.random.default_rng(11)
    original = rng.uniform(size=(64, 64, 3))
    dithered = original + rng.normal(scale=0.05, size=original.shape)
    frac = lowfreq_energy_fraction(dithered, original, channel=0)
    assert 0.0 <= frac <= 1.0


def test_lowfreq_energy_fraction_higher_for_low_frequency_drift():
    """Slow drift should register a higher low-frequency fraction than
    high-frequency noise of comparable magnitude."""
    height = width = 64
    rng = np.random.default_rng(12)
    original = np.zeros((height, width, 3))

    yy, xx = np.mgrid[0:height, 0:width]
    slow_drift = 0.05 * np.sin(2 * np.pi * xx / width)
    dithered_drift = original.copy()
    dithered_drift[..., 0] += slow_drift

    high_freq_noise = rng.choice([-0.05, 0.05], size=(height, width))
    dithered_noise = original.copy()
    dithered_noise[..., 0] += high_freq_noise

    frac_drift = lowfreq_energy_fraction(dithered_drift, original, channel=0)
    frac_noise = lowfreq_energy_fraction(dithered_noise, original, channel=0)
    assert frac_drift > frac_noise


def test_lowfreq_energy_fraction_zero_for_identical_images():
    """No error at all means no power to divide up; this should be 0.0, not NaN."""
    rng = np.random.default_rng(13)
    image = rng.uniform(size=(32, 32, 3))
    assert lowfreq_energy_fraction(image, image, channel=0) == 0.0


def test_spectral_anisotropy_zero_for_identical_images():
    rng = np.random.default_rng(14)
    image = rng.uniform(size=(32, 32, 3))
    assert spectral_anisotropy(image, image, channel=0) == 0.0


# --- S-CIELAB --------------------------------------------------------------- #


def test_scielab_de_identical_images():
    """S-CIELAB ΔE must be zero for identical images."""
    rng = np.random.default_rng(30)
    from tramage import srgb_to_oklab
    rgb = rng.uniform(0, 1, size=(64, 64, 3))
    img = srgb_to_oklab(rgb)
    assert scielab_de(img, img) == 0.0


def test_scielab_de_detects_noise():
    """S-CIELAB ΔE must be positive when the dithered image differs."""
    rng = np.random.default_rng(31)
    from tramage import srgb_to_oklab
    rgb = rng.uniform(0, 1, size=(64, 64, 3))
    img = srgb_to_oklab(rgb)
    noisy = img + rng.normal(scale=0.02, size=img.shape)
    de = scielab_de(noisy, img)
    assert de > 0.0


def test_scielab_de_monotonic_with_noise():
    """Larger noise should produce larger S-CIELAB ΔE."""
    rng = np.random.default_rng(32)
    from tramage import srgb_to_oklab
    rgb = rng.uniform(0, 1, size=(48, 48, 3))
    img = srgb_to_oklab(rgb)
    small = img + rng.normal(scale=0.01, size=img.shape)
    large = img + rng.normal(scale=0.05, size=img.shape)
    assert scielab_de(large, img) > scielab_de(small, img)


def test_scielab_de_full_returns_dict():
    """full=True must return a dict with mean, p95, and max keys."""
    rng = np.random.default_rng(33)
    from tramage import srgb_to_oklab
    rgb = rng.uniform(0, 1, size=(48, 48, 3))
    img = srgb_to_oklab(rgb)
    noisy = img + rng.normal(scale=0.03, size=img.shape)
    result = scielab_de(noisy, img, full=True)
    assert isinstance(result, dict)
    assert "mean" in result
    assert "p95" in result
    assert "max" in result
    assert result["mean"] <= result["p95"] <= result["max"]


def test_scielab_de_reduced_sensitivity_to_high_frequency():
    """S-CIELAB spatial filtering should make ΔE lower for high-frequency
    noise of the same pixel-level magnitude as low-frequency drift, because
    the contrast sensitivity function attenuates high frequencies."""
    rng = np.random.default_rng(34)
    from tramage import srgb_to_oklab
    height, width = 64, 64
    rgb = rng.uniform(0, 1, size=(height, width, 3))
    img = srgb_to_oklab(rgb)

    # Low-frequency: a slow sinusoidal modulation.
    yy, xx = np.mgrid[0:height, 0:width]
    slow = img.copy()
    slow[..., 0] += 0.05 * np.sin(2 * np.pi * xx / width)

    # High-frequency: pixel-alternating checkerboard of same amplitude.
    fast = img.copy()
    fast[..., 0] += 0.05 * np.where((xx + yy) % 2 == 0, 1.0, -1.0)

    de_slow = scielab_de(slow, img)
    de_fast = scielab_de(fast, img)
    # High-frequency noise should be perceptually less visible.
    assert de_fast < de_slow


def test_scielab_de_dpi_scaling():
    """Higher DPI (more pixels per visual degree) should narrow the spatial
    filters in visual angle terms, making the same pixel-level noise appear
    more visible (higher ΔE) because it occupies a larger visual angle."""
    rng = np.random.default_rng(35)
    from tramage import srgb_to_oklab
    rgb = rng.uniform(0, 1, size=(48, 48, 3))
    img = srgb_to_oklab(rgb)
    noisy = img + rng.normal(scale=0.03, size=img.shape)

    de_low_dpi = scielab_de(noisy, img, dpi=50, viewing_distance_cm=40)
    de_high_dpi = scielab_de(noisy, img, dpi=200, viewing_distance_cm=40)
    # At 200 DPI the same pixel grid subtends a smaller visual angle,
    # so the noise is finer and perceptually less visible.
    assert de_high_dpi < de_low_dpi


def test_scielab_de_flat_fields_match_cie76_both_filter_paths():
    """Flat colour pairs must return exactly CIE76: the filters are
    DC-preserving, so spatial filtering is identity on flat fields, in both
    the tabulated-kernel and Gaussian-approximation paths."""
    from tramage import srgb_to_cielab, srgb_to_oklab
    rgb0 = np.full((32, 40, 3), (0.8, 0.3, 0.5))
    rgb1 = np.full((32, 40, 3), (0.2, 0.6, 0.4))
    img0, img1 = srgb_to_oklab(rgb0), srgb_to_oklab(rgb1)
    expected = float(
        np.linalg.norm(srgb_to_cielab(rgb0)[0, 0] - srgb_to_cielab(rgb1)[0, 0])
    )
    for filters in ("zhang1997", "gaussian"):
        assert np.isclose(scielab_de(img1, img0, filters=filters), expected, rtol=1e-6)


def test_scielab_kernels_unit_dc_and_luminance_surround():
    """The tabulated kernels are DC-preserving; only the luminance kernel
    has a negative bandpass surround — the published signature of the
    Zhang & Wandell (1997) filters."""
    from tramage.metrics import _scielab_kernels
    lum, rg, by = _scielab_kernels(100.0, 40.0)
    for kernel in (lum, rg, by):
        assert np.isclose(kernel.sum(), 1.0)
    assert lum.min() < 0.0
    assert rg.min() >= 0.0 and by.min() >= 0.0
    # Support scales with each channel's widest lobe: 4.336° >> 0.494° > 0.386°.
    assert lum.shape[0] > rg.shape[0] > by.shape[0]


def test_scielab_zhang1997_matches_per_lobe_reference():
    """The FFT kernel path must agree with applying each published Gaussian
    lobe directly (unit-DC gaussian_filter per lobe, weighted sum) on the
    opponent image — an independent construction of the same filters."""
    from scipy.ndimage import gaussian_filter

    from tramage import srgb_to_oklab
    from tramage._constants import (
        SCIELAB_BY_SPREADS_DEG,
        SCIELAB_BY_WEIGHTS,
        SCIELAB_LUM_SPREADS_DEG,
        SCIELAB_LUM_WEIGHTS,
        SCIELAB_RG_SPREADS_DEG,
        SCIELAB_RG_WEIGHTS,
        pixels_per_degree,
    )
    from tramage.colour import oklab_to_xyz
    from tramage.metrics import _filter_opponent, _xyz_to_opponent

    rng = np.random.default_rng(40)
    img = srgb_to_oklab(rng.uniform(0, 1, size=(48, 48, 3)))
    opp = _xyz_to_opponent(oklab_to_xyz(img))
    ppd = pixels_per_degree(100.0, 40.0)
    reference = np.stack([
        sum(
            w * gaussian_filter(opp[..., c], spread * ppd)
            for w, spread in zip(weights, spreads)
        ) / sum(weights)
        for c, (weights, spreads) in enumerate((
            (SCIELAB_LUM_WEIGHTS, SCIELAB_LUM_SPREADS_DEG),
            (SCIELAB_RG_WEIGHTS, SCIELAB_RG_SPREADS_DEG),
            (SCIELAB_BY_WEIGHTS, SCIELAB_BY_SPREADS_DEG),
        ))
    ], axis=-1)
    got = _filter_opponent(opp, 100.0, 40.0, "zhang1997")
    # Agreement is to ~1e-6, not machine epsilon: the two paths truncate the
    # wide luminance lobe (sigma ~119 px on this 48 px image) differently at
    # the borders. A real plumbing error (axis swap, dropped lobe, bad
    # normalisation) would show up at O(0.01-1), orders above this bound.
    assert np.allclose(got, reference, atol=1e-5)


def test_scielab_de_unknown_filters_raises():
    import pytest

    from tramage import srgb_to_oklab
    img = srgb_to_oklab(np.full((8, 8, 3), 0.5))
    with pytest.raises(ValueError, match="unknown filters"):
        scielab_de(img, img, filters="boxcar")
