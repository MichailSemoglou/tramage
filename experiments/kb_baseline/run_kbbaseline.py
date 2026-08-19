"""Filter baseline experiment: shaping vs HVS-optimised filters.

Four active configurations per (image, K) — an FS/KB x baseline/shaped design:

- ``fs-baseline`` / ``fs-shaped``: Floyd-Steinberg, without / with shaping.
- ``kb-baseline`` / ``kb-shaped``: Kolpatzik-Bouman scalar HVS filter (the
  separable special case of Damera-Venkata), without / with shaping.

The FS/KB pair is a 2x2 factorial whose interaction is analysed in
``analyze_interaction.py``.

The matrix-valued Damera-Venkata filter (``dv-baseline`` / ``dv-shaped``) is
disabled pending a design fix: 0 clip events, an exact opponent round-trip,
and DC-conserving taps, yet high bias and low-frequency error on real images
(see ROADMAP). Set ``RUN_DV = True`` to re-enable.

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

from tramage._harness import (
    corpus_dirs,
    finalise_csv,
    images_in,
    load_done,
    load_oklab,
    provenance,
    score,
)
from tramage.diffusion import diffuse
from tramage.dv_filter import dv_diffuse, dv_filters
from tramage.kb_filter import kb_diffuse
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
    "tone_error", "lowfreq_frac", "anisotropy", "scielab_de",
    "git_sha", "corpus_sha256",
]

# DV (matrix-valued Damera-Venkata) is disabled until its design is fixed:
# 0 clip events, exact opponent round-trip, DC-conserving taps, clean
# constant-image output, yet high bias/lowfreq on real images (see ROADMAP).
RUN_DV = False
CONFIGS = (
    "fs-baseline", "fs-shaped", "kb-baseline", "kb-shaped",
) + (("dv-baseline", "dv-shaped") if RUN_DV else ())
_CONFIG_ORDER = {c: i for i, c in enumerate(CONFIGS)}


def _done_key(row: dict) -> tuple:
    """Identity of one trial for resume-skipping: (image, K, config)."""
    return (row["image"], int(row["K"]), row["config"])


def _row_key(row: dict) -> tuple:
    """Identity of one CSV row for dedupe."""
    return (row["corpus"], row["image"], int(row["K"]), row["config"])


def _sort_key(row: dict) -> tuple:
    """Sort key putting configs in their declared order within each (image, K)."""
    return (row["corpus"], row["image"], int(row["K"]), _CONFIG_ORDER.get(row["config"], 99))


def _run_config(
    oklab: np.ndarray,
    palette: np.ndarray,
    M: np.ndarray,
    config: str,
    dv_filts=None,
) -> np.ndarray:
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
    if config == "dv-baseline":
        d, _ = dv_diffuse(oklab, palette, filters=dv_filts)
        return d
    if config == "dv-shaped":
        d, _ = dv_diffuse(oklab, palette, filters=dv_filts, shaping=M)
        return d
    raise ValueError(f"unknown config {config!r}")


def _timing_probe(images: list[Path], n: int | None = None) -> float:
    configs = list(CONFIGS)
    if n is None:
        n = len(configs)
    print(f"timing probe: {n} dithers...", flush=True)
    times: list[float] = []
    for i in range(n):
        oklab = load_oklab(images[i % len(images)])
        palette, _ = extract_palette(oklab, n_colours=16, seed=SEED)
        M, _, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
        dv_filts = dv_filters(palette) if RUN_DV else None
        t0 = time.time()
        _run_config(oklab, palette, M, configs[i], dv_filts)
        times.append(time.time() - t0)
        print(f"  probe {i+1}: {configs[i]}  {times[-1]:.2f}s", flush=True)
    mean = sum(times) / len(times)
    print(f"mean per-trial: {mean:.2f}s  -> 864 trials est. {mean*864/60:.1f} min", flush=True)
    return mean


def main() -> None:
    corpora = [d for d in corpus_dirs(CORPORA) if d.name == 'kodak']
    if not corpora:
        raise SystemExit(
            f"no corpus directories found under {CORPORA}. "
            "Run experiments/palette_geometry/fetch_kodak.py first."
        )

    done = load_done(CSV_PATH, _done_key)
    if done:
        print(f"resuming: {len(done)} trials already in CSV, will skip those", flush=True)

    all_images: list[Path] = []
    for corpus in corpora:
        all_images += images_in(corpus)
    if not all_images:
        raise SystemExit("no .png/.jpg images found in any corpus")

    prov = provenance(all_images, HERE.parent.parent)

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
            images = images_in(corpus)
            if not images:
                continue
            print(f"[corpus] {corpus.name}: {len(images)} images", flush=True)

            for path in images:
                oklab = load_oklab(path)
                for K in KS:
                    if all((path.name, K, c) in done for c in CONFIGS):
                        continue
                    palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
                    M, _, _ = global_shaping_matrix(palette, eps=DEFAULT_EPS)
                    dv_filts = None
                    if RUN_DV and any((path.name, K, c) not in done
                                      for c in ("dv-baseline", "dv-shaped")):
                        dv_filts = dv_filters(palette)
                    for config in CONFIGS:
                        if (path.name, K, config) in done:
                            continue
                        dithered = _run_config(oklab, palette, M, config, dv_filts)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "config": config, "eps": DEFAULT_EPS,
                        }
                        row.update(score(dithered, oklab))
                        row.update(prov)
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done {len(KS)} K x {len(CONFIGS)} configs "
                      f"({n_new + len(done)} total)", flush=True)
    finally:
        fh.close()

    finalise_csv(CSV_PATH, FIELDS, _row_key, _sort_key)
    print(f"\n{n_new} new trial(s) added; {n_new + len(done)} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
