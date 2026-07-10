"""KB baseline experiment: does shaping help against a stronger filter?

Runs four configurations on each (image, K):

- ``fs-baseline``:  Floyd-Steinberg in Oklab, no shaping.
- ``fs-shaped``:    FS in Oklab with global palette shaping.
- ``kb-baseline``:  Kolpatzik-Bouman scalar filter in opponent space, no shaping.
- ``kb-shaped``:    KB filter with global palette shaping applied on top.

The fourth configuration is the load-bearing one. If ``kb-shaped`` beats
``kb-baseline``, the paper's claim generalises beyond plain FS. If not,
the claim is FS-specific and the paper narrows.

Crash-safe (append-per-trial, dedupe at end). Same pattern as the other
harnesses.

Run from the repository root::

    python experiments/kb_baseline/run_kbbaseline.py
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
from PIL import Image

from tramage.colour import srgb_to_oklab
from tramage.diffusion import diffuse
from tramage.kb_filter import kb_diffuse, kb_filters
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
CSV_PATH = RESULTS / "kbbaseline.csv"

FIELDS = [
    "corpus", "image", "K", "config", "eps",
    "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b",
    "tone_error", "lowfreq_frac", "anisotropy",
]

CONFIGS = ("fs-baseline", "fs-shaped", "kb-baseline", "kb-shaped")


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
    order = {c: i for i, c in enumerate(CONFIGS)}
    for row in rows:
        key = (row["corpus"], row["image"], int(row["K"]), row["config"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda r: (r["corpus"], r["image"], int(r["K"]),
                               order.get(r["config"], 99)))
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


def _load_oklab(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    return srgb_to_oklab(rgb)


def _run_config(oklab: np.ndarray, palette: np.ndarray, M: np.ndarray, config: str) -> np.ndarray:
    """Return the dithered image for one configuration."""
    if config == "fs-baseline":
        d, _ = diffuse(oklab, palette, shaping=None)
        return d
    if config == "fs-shaped":
        d, _ = diffuse(oklab, palette, shaping=M)
        return d
    if config == "kb-baseline":
        d, _ = kb_diffuse(oklab, palette)
        return d
    if config == "kb-shaped":
        d, _ = kb_diffuse(oklab, palette, shaping=M)
        return d
    raise ValueError(f"unknown config {config!r}")


def _timing_probe(images: list[Path], n: int = 4) -> float:
    print(f"timing probe: {n} dithers...", flush=True)
    configs = ["fs-baseline", "kb-baseline", "fs-shaped", "kb-shaped"]
    times: list[float] = []
    for i in range(n):
        oklab = _load_oklab(images[i % len(images)])
        palette, _ = extract_palette(oklab, n_colours=16, seed=SEED)
        M, _, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
        t0 = time.time()
        _run_config(oklab, palette, M, configs[i])
        times.append(time.time() - t0)
        print(f"  probe {i+1}: {configs[i]}  {times[-1]:.2f}s", flush=True)
    mean = sum(times) / len(times)
    print(f"mean per-trial: {mean:.2f}s  -> 576 trials est. {mean*576/60:.1f} min", flush=True)
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
                oklab = _load_oklab(path)
                for K in KS:
                    if all((path.name, K, c) in done for c in CONFIGS):
                        continue
                    palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
                    M, _, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
                    for config in CONFIGS:
                        if (path.name, K, config) in done:
                            continue
                        dithered = _run_config(oklab, palette, M, config)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "config": config, "eps": DEFAULT_EPS,
                        }
                        row.update(_score(dithered, oklab))
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
