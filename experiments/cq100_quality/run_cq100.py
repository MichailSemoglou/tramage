"""Dither the CQ100 corpus with three variants and score each on fidelity/noise.

Mirrors ``experiments/dithering_quality/run_quality.py`` but operates on the
CQ100 corpus (100 images, 8 categories). CQ100 provides category-level
breakdowns that Kodak24 cannot: animals, food, objects, people, places,
plants, vehicles, miscellaneous. The image-to-category mapping lives in
``experiments/corpora/cq100/categories.csv`` (hand-prepared, one category
per image); when it is absent, every row gets an empty category.

Each (image, K) trial is scored with ``tramage.metrics``. Output is one
row per (corpus, image, K, variant) in a CSV. Crash-safe: each row is
appended and flushed immediately.

Run from the repository root::

    python experiments/cq100_quality/run_cq100.py

Dataset: download CQ100 from https://data.mendeley.com/datasets/vw5ys9hfxw
(CC BY 4.0) and extract to ``experiments/corpora/cq100/``.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

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
from tramage.palette import extract_palette
from tramage.shaping import DEFAULT_EPS, global_shaping_matrix, per_cell_shaping_matrices

KS: tuple[int, ...] = (8, 16, 32, 64, 128, 256)
SEED = 0

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent / "corpora"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "cq100_dithering_quality.csv"

FIELDS = [
    "corpus", "image", "category", "K", "variant", "eps",
    "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b",
    "tone_error", "lowfreq_frac", "anisotropy", "scielab_de",
    "git_sha", "corpus_sha256",
]


def _done_key(row: dict) -> tuple:
    """Identity of one trial for resume-skipping: (image, K, variant)."""
    return (row["image"], int(row["K"]), row["variant"])


def _row_key(row: dict) -> tuple:
    """Identity of one CSV row for dedupe and sort."""
    return (row["corpus"], row["image"], int(row["K"]), row["variant"])


def _timing_probe(images: list[Path], n: int = 5) -> float:
    print(f"timing probe: {n} dithers across mixed K values...", flush=True)
    probe_K = [8, 32, 128, 256, 256][:n]
    times: list[float] = []
    for i, K in enumerate(probe_K):
        oklab = load_oklab(images[i % len(images)])
        palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
        t0 = time.time()
        diffuse(oklab, palette, shaping=None)
        dt = time.time() - t0
        times.append(dt)
        print(f"  probe {i+1}: K={K}  {dt:.2f}s", flush=True)
    mean = sum(times) / len(times)
    trials = len(images) * len(KS) * 3
    print(
        f"mean per-dither: {mean:.2f}s  ->  {trials} trials est. "
        f"{mean*trials/60:.1f} min",
        flush=True,
    )
    return mean


def main() -> None:
    corpora = [d for d in corpus_dirs(CORPORA) if d.name == 'cq100']
    if not corpora:
        raise SystemExit(
            f"no corpus directories found under {CORPORA}. "
            "Download CQ100 from https://data.mendeley.com/datasets/vw5ys9hfxw "
            "and extract to experiments/corpora/cq100/"
        )

    done = load_done(CSV_PATH, _done_key)
    if done:
        print(f"resuming: {len(done)} trials already in CSV, will skip those", flush=True)

    all_images: list[Path] = []
    for corpus in corpora:
        all_images += images_in(corpus)
    if not all_images:
        raise SystemExit("no .png/.jpg/.ppm images found in any corpus")

    prov = provenance(all_images, HERE.parent.parent)

    _timing_probe(all_images, n=5)

    categories: dict[str, str] = {}
    cat_csv = CORPORA / "cq100" / "categories.csv"
    if cat_csv.exists():
        with cat_csv.open(newline="") as fh:
            for row in csv.DictReader(fh):
                categories[row["image"]] = row["category"]

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
                            "category": categories.get(path.name, ""),
                            "K": K, "variant": variant, "eps": DEFAULT_EPS,
                        }
                        row.update(score(dithered, oklab))
                        row.update(prov)
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done {len(KS)} K x 3 variants "
                      f"({n_new + len(done)} total)", flush=True)
    finally:
        fh.close()

    finalise_csv(CSV_PATH, FIELDS, _row_key)
    print(f"\n{n_new} new trial(s) added; {n_new + len(done)} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
