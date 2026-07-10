"""Dither the Kodak corpus with three variants and score each on fidelity/noise.

The geometry check in ``experiments/palette_geometry`` established that
k-means palettes extracted from natural images are consistently and
substantially anisotropic in Oklab. This script asks the load-bearing
follow-up: does that anisotropy translate into measurable dithering
improvement when the shaping matrix is actually applied during diffusion?

Three variants are compared on every (image, K) trial:

- ``baseline``: ``shaping=None`` (vector Floyd-Steinberg).
- ``global``:   one shaping matrix from the whole palette's covariance.
- ``cell``:     one shaping matrix per palette colour, from its k nearest
                neighbours.

Each dither is scored with ``tramage.metrics``. No new method code; this
exercises the package's public API unchanged. Output is one row per
(image, K, variant) in a CSV.

The script is crash-safe: each row is appended and flushed as soon as it
is computed, so an interruption loses at most the trial in flight. A
5-trial timing probe runs first, so the full run's cost is known before
the whole job commits.

Run from the repository root, after ``palette_geometry/fetch_kodak.py``::

    python experiments/dithering_quality/run_quality.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tramage.colour import srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.metrics import (
    blurred_tone_error,
    lowfreq_energy_fraction,
    noise_std,
    spectral_anisotropy,
    systematic_bias,
)
from tramage.palette import extract_palette
from tramage.shaping import DEFAULT_EPS, global_shaping_matrix, per_cell_shaping_matrices

# Same K values as the geometry experiment, so the two CSVs join cleanly
# on (image, K) for the gain-vs-geometry scatter.
KS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
SEED = 0  # Single seed; palette-extraction variance was covered by the geometry run.

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent / "corpora"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "dithering_quality.csv"

FIELDS = [
    "corpus", "image", "K", "variant", "eps",
    "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b",
    "tone_error", "lowfreq_frac", "anisotropy",
]


def _load_done() -> set[tuple[str, int, str]]:
    """Return the set of ``(image, K, variant)`` trials already in the CSV."""
    if not CSV_PATH.exists():
        return set()
    done: set[tuple[str, int, str]] = set()
    with CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["image"], int(row["K"]), row["variant"]))
    return done


def _finalise_csv() -> None:
    """Dedupe and sort. Idempotent; safe to call repeatedly."""
    if not CSV_PATH.exists():
        return
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["corpus"], row["image"], int(row["K"]), row["variant"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda r: (r["corpus"], r["image"], int(r["K"]), r["variant"]))
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique)


def _load_oklab(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    return srgb_to_oklab(rgb)


def _score(dithered: np.ndarray, original: np.ndarray) -> dict:
    """Compute every metric for one dithered image against its original.

    Returns the metric fields of one CSV row (caller adds identity fields).
    """
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


def _timing_probe(images: list[Path], n: int = 5) -> float:
    """Time `n` full dithers across a spread of K values; return mean seconds."""
    print(f"timing probe: {n} dithers across mixed K values...", flush=True)
    probe_K = [8, 32, 128, 256, 256][:n]
    times: list[float] = []
    for i, K in enumerate(probe_K):
        oklab = _load_oklab(images[i % len(images)])
        palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
        t0 = time.time()
        diffuse(oklab, palette, shaping=None)
        dt = time.time() - t0
        times.append(dt)
        print(f"  probe {i+1}: K={K}  {dt:.2f}s", flush=True)
    mean = sum(times) / len(times)
    print(f"mean per-dither: {mean:.2f}s  -> 432 trials est. {mean*432/60:.1f} min", flush=True)
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

    # Gather all images once; the probe needs them too.
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
                oklab = _load_oklab(path)
                for K in KS:
                    if all((path.name, K, v) in done for v in ("baseline", "global", "cell")):
                        continue
                    palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
                    M_global, _, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
                    M_cell, _, _ = per_cell_shaping_matrices(palette, k=4, eps=DEFAULT_EPS)

                    for variant, shaping in (
                        ("baseline", None),
                        ("global", M_global),
                        ("cell", M_cell),
                    ):
                        if (path.name, K, variant) in done:
                            continue
                        dithered, _ = diffuse(oklab, palette, shaping=shaping)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "variant": variant, "eps": DEFAULT_EPS,
                        }
                        row.update(_score(dithered, oklab))
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done {len(KS)} K x 3 variants "
                      f"({n_new + len(done)} total)", flush=True)
    finally:
        fh.close()

    _finalise_csv()
    print(f"\n{n_new} new trial(s) added; {n_new + len(done)} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
