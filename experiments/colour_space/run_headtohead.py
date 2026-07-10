"""Oklab-vs-CIELAB head-to-head: which space gives shaping more purchase?

The package defaults to Oklab for palette-structure-aware shaping, on the
assertion that Oklab's superior perceptual uniformity makes its covariance
geometry more meaningful. This script tests that assertion directly by
running the same Kodak24 dithering pipeline in three configurations:

- ``oklab``:   shape in Oklab, score in Oklab.
- ``cielab``:  shape in CIELAB, score in CIELAB.
- ``baseline``: no shaping, in Oklab (the vector Floyd-Steinberg reference).

The metrics are computed in the colour space the dithering was done in, so
absolute values are not cross-comparable (CIELAB L* is 0-100, Oklab L is
~0-1). The comparison that matters is the **relative shaping improvement**
within each space, computed downstream in ``plot_headtohead.py`` as
``(metric_baseline - metric_shaped) / metric_baseline``. That ratio is
unit-free and directly answers "which space gives shaping more purchase?"

Per-cell shaping is intentionally not tested here: the
``experiments/dithering_quality/`` corpus run found it does not earn its
complexity, so only the global variant is compared across spaces.

Crash-safe (append-per-trial, dedupe at end). Same pattern as the other
two experiment harnesses.

Run from the repository root::

    python experiments/colour_space/run_headtohead.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tramage.colour import cielab_to_srgb, srgb_to_cielab, srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.metrics import (
    blurred_tone_error,
    lowfreq_energy_fraction,
    noise_std,
    spectral_anisotropy,
    systematic_bias,
)
from tramage.palette import extract_palette
from tramage.shaping import DEFAULT_EPS, global_shaping_matrix

KS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
SEED = 0

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent / "corpora"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "headtohead.csv"

FIELDS = [
    "corpus", "image", "K", "config", "eps",
    "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b",
    "tone_error", "lowfreq_frac", "anisotropy",
]

# Each configuration is a self-contained pipeline: a forward transform,
# whether to shape, and (implicitly) that scoring happens in the same space.
# Two baselines (one per space) so the relative-improvement comparison is
# coherent within each space: oklab shaping is compared against the Oklab
# baseline, cielab shaping against the CIELAB baseline. Comparing across
# spaces directly would mix Oklab L (~0-1) with CIELAB L* (0-100).
CONFIGS = ("oklab-baseline", "oklab", "cielab-baseline", "cielab")


def _load_done() -> set[tuple[str, int, str]]:
    if not CSV_PATH.exists():
        return set()
    done: set[tuple[str, int, str]] = set()
    with CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["image"], int(row["K"]), row["config"]))
    return done


def _finalise_csv() -> None:
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["corpus"], row["image"], int(row["K"]), row["config"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda r: (r["corpus"], r["image"], int(r["K"]),
                               {"oklab-baseline": 0, "oklab": 1,
                                "cielab-baseline": 2, "cielab": 3}[r["config"]]))
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique)


def _score(dithered: np.ndarray, original: np.ndarray) -> dict:
    bias = systematic_bias(dithered, original)
    noise = noise_std(dithered, original)
    return {
        "bias_norm": float(np.linalg.norm(bias)),
        "bias_L": float(bias[0]), "bias_a": float(bias[1]), "bias_b": float(bias[2]),
        "noise_L": float(noise[0]), "noise_a": float(noise[1]), "noise_b": float(noise[2]),
        "tone_error": float(blurred_tone_error(dithered, original)),
        "lowfreq_frac": float(lowfreq_energy_fraction(dithered, original, channel=0)),
        "anisotropy": float(spectral_anisotropy(dithered, original, channel=0)),
    }


def _load_rgb(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.float64) / 255.0


def _space_relative_eps(palette: np.ndarray) -> float:
    """Set eps as a fixed fraction of the palette's largest eigenvalue.

    The package default ``DEFAULT_EPS = 0.02**2`` is tuned for Oklab's
    scale (L ~ 0-1). CIELAB L* ranges 0-100, so its eigenvalues are ~10,000x
    larger, and a fixed eps would give every gain as ~1.0 (no shaping).
    Setting eps relative to the palette's own scale makes the gain
    threshold comparable across spaces: an axis is "flat" if its variance
    is below the same *fraction* of the dominant axis, regardless of units.

    The fraction 0.0004/0.05 ~= 0.8% is calibrated so that on a typical
    Kodak Oklab palette, this reproduces the package default's behaviour
    (the smallest eigenvalue gets a gain near 0.15, the largest near 0.99).
    """
    _, eigenvalues, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
    return float(eigenvalues[-1]) * (DEFAULT_EPS / 0.05)


def _run_config(rgb: np.ndarray, K: int, config: str) -> tuple[np.ndarray, np.ndarray, float]:
    """Run one configuration end-to-end; return (dithered, original, eps_used).

    The third return value is the eps actually used for shaping, so it can
    be recorded in the CSV for transparency. For the baseline configs it is
    the package default (no shaping occurs, so the value is informational).
    """
    if config == "oklab-baseline":
        img = srgb_to_oklab(rgb)
        palette, _ = extract_palette(img, n_colours=K, seed=SEED)
        dithered, _ = diffuse(img, palette, shaping=None)
        return dithered, img, DEFAULT_EPS
    if config == "oklab":
        img = srgb_to_oklab(rgb)
        palette, _ = extract_palette(img, n_colours=K, seed=SEED)
        eps = _space_relative_eps(palette)
        M, _, _ = global_shaping_matrix(palette, eps=eps)
        dithered, _ = diffuse(img, palette, shaping=M)
        return dithered, img, eps
    if config == "cielab-baseline":
        img = srgb_to_cielab(rgb)
        palette, _ = extract_palette(img, n_colours=K, seed=SEED)
        dithered, _ = diffuse(img, palette, shaping=None)
        return dithered, img, DEFAULT_EPS
    if config == "cielab":
        img = srgb_to_cielab(rgb)
        palette, _ = extract_palette(img, n_colours=K, seed=SEED)
        eps = _space_relative_eps(palette)
        M, _, _ = global_shaping_matrix(palette, eps=eps)
        dithered, _ = diffuse(img, palette, shaping=M)
        return dithered, img, eps
    raise ValueError(f"unknown config {config!r}")


def _timing_probe(images: list[Path], n: int = 3) -> float:
    print(f"timing probe: {n} full pipelines...", flush=True)
    times: list[float] = []
    configs = ["oklab-baseline", "oklab", "cielab-baseline", "cielab"]
    for i in range(n):
        rgb = _load_rgb(images[i % len(images)])
        K = [8, 64, 256][i]
        t0 = time.time()
        d, o, _ = _run_config(rgb, K, configs[i])
        _score(d, o)
        dt = time.time() - t0
        times.append(dt)
        print(f"  probe {i+1}: {configs[i]} K={K}  {dt:.2f}s", flush=True)
    mean = sum(times) / len(times)
    print(f"mean per-trial: {mean:.2f}s  -> 432 trials est. {mean*432/60:.1f} min", flush=True)
    return mean


def main() -> None:
    corpora = sorted(d for d in CORPORA.iterdir() if d.is_dir())
    if not corpora:
        raise SystemExit(
            f"no corpus directories found under {CORPORA}. "
            "Run experiments/palette_geometry/fetch_kodak.py first."
        )

    done = _load_done()
    if done:
        print(f"resuming: {len(done)} trials already in CSV, will skip those", flush=True)

    all_images: list[Path] = []
    for corpus in corpora:
        all_images += sorted(corpus.glob("*.png")) + sorted(corpus.glob("*.jpg"))
    if not all_images:
        raise SystemExit("no .png/.jpg images found in any corpus")

    _timing_probe(all_images)

    is_new_file = not CSV_PATH.exists()
    fh = CSV_PATH.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if is_new_file:
        writer.writeheader()
        fh.flush()

    n_new = 0
    try:
        for corpus in corpora:
            images = sorted(corpus.glob("*.png")) + sorted(corpus.glob("*.jpg"))
            if not images:
                continue
            print(f"[corpus] {corpus.name}: {len(images)} images", flush=True)

            for path in images:
                rgb = _load_rgb(path)
                for K in KS:
                    if all((path.name, K, c) in done for c in CONFIGS):
                        continue
                    for config in CONFIGS:
                        if (path.name, K, config) in done:
                            continue
                        dithered, original, eps_used = _run_config(rgb, K, config)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "config": config, "eps": eps_used,
                        }
                        row.update(_score(dithered, original))
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done {len(KS)} K x {len(CONFIGS)} configs "
                      f"({n_new + len(done)} total)", flush=True)
    finally:
        fh.close()

    _finalise_csv()
    print(f"\n{n_new} new trial(s) added; {n_new + len(done)} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
