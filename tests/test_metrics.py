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
