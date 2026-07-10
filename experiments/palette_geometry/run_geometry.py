"""Extract k-means palettes from a corpus and report their Oklab geometry.

For each image x K x seed, this script extracts a k-means palette in Oklab
and records the eigenstructure that :mod:`tramage.shaping` would act on, so
the Phase 1 reframing question -- "are extracted palettes consistently
near-degenerate in Oklab, or is near-degeneracy the exception?" -- can be
answered from data rather than assertion.

No new method code is exercised here. The functions called are the package's
public API: :func:`tramage.palette.extract_palette` and
:func:`tramage.palette.eigenspectrum_report`, with the same classification
bands that :func:`tramage.palette.describe_palette` uses for plain-language
reporting. Output is one row per (image, K, seed) trial in a CSV.

Run from the repository root::

    python experiments/palette_geometry/run_geometry.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

from tramage.colour import srgb_to_oklab
from tramage.palette import _FLAT_GAIN, _REAL_EXTENT_GAIN, eigenspectrum_report, extract_palette
from tramage.shaping import DEFAULT_EPS

# K values span the small-palette regime where shaping is hypothesised to
# matter (designer-palette territory: riso, duotone, brand sets) through to
# the large-palette regime where classical CQ typically lives. The
# reframing question is precisely whether near-degeneracy survives as K
# grows.
KS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
SEEDS: tuple[int, ...] = (0, 1, 2)

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent / "corpora"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "palette_geometry.csv"

FIELDS = [
    "corpus", "image", "K", "seed", "eps",
    "lambda1", "lambda2", "lambda3",
    "g1", "g2", "g3",
    "n_flat", "n_partial", "n_real",
]


def _classify(gains: np.ndarray) -> tuple[int, int, int]:
    """Count axes in each band, using the package's own thresholds.

    Consistent with :func:`tramage.palette.describe_palette`: flat is
    ``gain <= _FLAT_GAIN`` (0.3), real extent is ``>= _REAL_EXTENT_GAIN``
    (0.7), partial is in between.
    """
    n_flat = int(np.sum(gains <= _FLAT_GAIN))
    n_real = int(np.sum(gains >= _REAL_EXTENT_GAIN))
    n_partial = len(gains) - n_flat - n_real
    return n_flat, n_partial, n_real


def _load_oklab(path: Path) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    rgb = np.asarray(img, dtype=np.float64) / 255.0
    return srgb_to_oklab(rgb)


def _load_done() -> tuple[set[tuple[str, int, int]], list[dict]]:
    """Return the trials already in the CSV and the full existing rows.

    Lets the run resume after an interruption. The larger K values are
    expensive (a 256-colour k-means on a 768x512 image takes ~10 s), and
    a fresh run each time would throw away that work; the run is also
    short enough that interrupting and resuming later is a realistic
    workflow, so progress is checkpointed per trial rather than buffered.
    """
    if not CSV_PATH.exists():
        return set(), []
    done: set[tuple[str, int, int]] = set()
    rows: list[dict] = []
    with CSV_PATH.open(newline="") as fh:
        for row in csv.DictReader(fh):
            done.add((row["image"], int(row["K"]), int(row["seed"])))
            rows.append(row)
    return done, rows


def _finalise_csv() -> None:
    """Dedupe (in case the same trial was written twice across runs) and sort.

    Idempotent; safe to call repeatedly. Sort order matches the run order
    so successive runs produce a byte-identical CSV.
    """
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen: set[tuple] = set()
    unique: list[dict] = []
    for row in rows:
        key = (row["corpus"], row["image"], int(row["K"]), int(row["seed"]))
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
    unique.sort(key=lambda r: (r["corpus"], r["image"], int(r["K"]), int(r["seed"])))
    with CSV_PATH.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(unique)


def main() -> None:
    corpora = sorted(d for d in CORPORA.iterdir() if d.is_dir())
    if not corpora:
        raise SystemExit(
            f"no corpus directories found under {CORPORA}. "
            "Run fetch_kodak.py first, or place images in a subdirectory of "
            "experiments/corpora/."
        )

    done, existing = _load_done()
    n_existing = len(existing)
    if n_existing:
        print(f"resuming: {n_existing} trials already in CSV, will skip those", flush=True)

    # Open in append mode and write each row as soon as it is computed, so
    # an interruption (timeout, kill, crash) loses at most the single trial
    # in flight, not the whole run. Dedup happens once at the end.
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
                print(f"[skip] {corpus.name}: no .png/.jpg images", flush=True)
                continue
            print(f"[corpus] {corpus.name}: {len(images)} images", flush=True)

            for path in images:
                already = sum(1 for K in KS for seed in SEEDS if (path.name, K, seed) in done)
                if already == len(KS) * len(SEEDS):
                    print(f"  {path.name}: already complete, skipping", flush=True)
                    continue
                oklab = _load_oklab(path)
                for K in KS:
                    for seed in SEEDS:
                        if (path.name, K, seed) in done:
                            continue
                        palette, _ = extract_palette(oklab, n_colours=K, seed=seed)
                        report = eigenspectrum_report(palette, eps=DEFAULT_EPS)
                        n_flat, n_partial, n_real = _classify(report.gains)
                        writer.writerow({
                            "corpus": corpus.name,
                            "image": path.name,
                            "K": K,
                            "seed": seed,
                            "eps": DEFAULT_EPS,
                            "lambda1": float(report.eigenvalues[0]),
                            "lambda2": float(report.eigenvalues[1]),
                            "lambda3": float(report.eigenvalues[2]),
                            "g1": float(report.gains[0]),
                            "g2": float(report.gains[1]),
                            "g3": float(report.gains[2]),
                            "n_flat": n_flat,
                            "n_partial": n_partial,
                            "n_real": n_real,
                        })
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done {len(KS)} K x {len(SEEDS)} seeds "
                      f"({n_new + n_existing} total)", flush=True)
    finally:
        fh.close()

    _finalise_csv()
    print(f"\n{n_new} new trial(s) added; {n_new + n_existing} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
