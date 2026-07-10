"""Sweep epsilon across orders of magnitude and score each on fidelity/noise.

The epsilon parameter controls the variance-scale below which an axis is
treated as degenerate: larger eps damps more aggressively (less noise, less
off-plane fidelity), smaller eps approaches the unshaped baseline.

This script fixes one image and palette, sweeps eps from 1e-6 to 1e-1
(log-spaced), and scores each trial on:

- noise_std (L channel)
- blurred_tone_error
- scielab_de (S-CIELAB ΔE, the field's standard perceptual metric)

Output is a CSV in ``results/epsilon_sweep.csv``, one row per eps value.
The purpose is to produce the noise-vs-tone Pareto front that turns eps
from a tunable parameter into a characterised trade-off.

Run from the repository root::

    python experiments/epsilon_sweep/run_sweep.py [--image IMAGE] [--K K] [--seed SEED]
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image

from tramage.colour import oklab_to_srgb, srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.metrics import blurred_tone_error, noise_std, scielab_de
from tramage.palette import extract_palette
from tramage.shaping import DEFAULT_EPS, global_shaping_matrix

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)

EPS_BOTTOM = 1e-6
EPS_TOP = 1e-1
N_EPS = 40

FIELDS = ["eps", "noise_L", "tone_error", "scielab_de"]


def _make_test_image(height: int = 256, width: int = 256) -> np.ndarray:
    """A synthetic gradient with an off-axis blob, same design as gallery."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float64)
    xn, yn = xx / (width - 1), yy / (height - 1)
    img = np.zeros((height, width, 3))
    img[..., 0] = 0.40 + 0.30 * xn
    img[..., 1] = -0.08 + 0.16 * yn
    blob = np.exp(-(((xn - 0.5) ** 2 + (yn - 0.5) ** 2) / (2 * 0.18 ** 2)))
    img[..., 2] = 0.02 + 0.09 * blob
    return srgb_to_oklab(np.clip(oklab_to_srgb(img), 0, 1))


def _load_image(path: Path | None) -> np.ndarray:
    if path is None:
        return _make_test_image()
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float64) / 255.0
    return srgb_to_oklab(rgb)


def main() -> None:
    parser = argparse.ArgumentParser(description="Epsilon sweep for palette shaping.")
    parser.add_argument("--image", type=Path, default=None,
                        help="Path to an image file. Defaults to a synthetic gradient.")
    parser.add_argument("--K", type=int, default=8,
                        help="Number of palette colours to extract (default: 8).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for k-means (default: 0).")
    args = parser.parse_args()

    image = _load_image(args.image)
    palette, _ = extract_palette(image, n_colours=args.K, seed=args.seed)

    csv_path = RESULTS / "epsilon_sweep.csv"
    epsilons = np.logspace(np.log10(EPS_BOTTOM), np.log10(EPS_TOP), N_EPS)

    print(f"image: {args.image or 'synthetic gradient'}")
    print(f"K: {args.K}, seed: {args.seed}")
    print(f"eps range: {EPS_BOTTOM:.0e} – {EPS_TOP:.0e} ({N_EPS} steps)")
    print(f"default eps (DEFAULT_EPS): {DEFAULT_EPS:.2e}")
    print()

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()

        for i, eps in enumerate(epsilons):
            M, _, _ = global_shaping_matrix(palette, eps=eps)
            dithered, _ = diffuse(image, palette, shaping=M)

            row = {
                "eps": eps,
                "noise_L": float(noise_std(dithered, image)[0]),
                "tone_error": float(blurred_tone_error(dithered, image)),
                "scielab_de": float(scielab_de(dithered, image)),
            }
            writer.writerow(row)
            fh.flush()
            print(f"  [{i+1:2d}/{N_EPS}] eps={eps:.2e}  "
                  f"noise_L={row['noise_L']:.4f}  "
                  f"tone_error={row['tone_error']:.4f}  "
                  f"scielab_de={row['scielab_de']:.2f}")

    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
