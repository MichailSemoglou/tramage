"""Corpus-scale epsilon sensitivity for the shaping matrix.

`run_sweep.py` characterizes eps on a single synthetic image. This script
asks the corpus-scale question the paper needs: how much do the headline
numbers move when eps leaves its default? For every (Kodak image, K) it
dithers once unshaped and once per eps multiplier, scoring each dither with
the shared harness metrics (S-CIELAB included).

Design: Kodak24 x K in {8, 16, 32} x eps in {0.25, 0.5, 1, 2, 4} x
DEFAULT_EPS (0.02^2 = 4e-4). The baseline dither does not depend on eps and
is computed once per (image, K). K stops at 32: the subgroup analysis shows
the S-CIELAB benefit is a small-K phenomenon, and K=8-32 is the regime the
paper claims.

Crash-safe (append-per-trial, dedupe at end). Run from the repository root::

    python experiments/epsilon_sweep/run_corpus_sweep.py
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
from tramage.shaping import DEFAULT_EPS, global_shaping_matrix

KS: tuple[int, ...] = (8, 16, 32)
EPS_MULTS: tuple[float, ...] = (0.25, 0.5, 1.0, 2.0, 4.0)
SEED = 0

HERE = Path(__file__).resolve().parent
CORPORA = HERE.parent / "corpora"
RESULTS = HERE / "results"
RESULTS.mkdir(exist_ok=True)
CSV_PATH = RESULTS / "corpus_sweep.csv"

FIELDS = [
    "corpus", "image", "K", "variant", "eps_mult", "eps",
    "bias_norm", "bias_L", "bias_a", "bias_b",
    "noise_L", "noise_a", "noise_b",
    "tone_error", "lowfreq_frac", "anisotropy", "scielab_de",
    "git_sha", "corpus_sha256",
]


def _done_key(row: dict) -> tuple:
    """Identity of one trial for resume-skipping: (image, K, eps_mult)."""
    return (row["image"], int(row["K"]), row["eps_mult"])


def _row_key(row: dict) -> tuple:
    """Identity of one CSV row for dedupe and sort.

    Baseline rows carry eps_mult '' and sort first within each (image, K).
    """
    mult = float(row["eps_mult"]) if row["eps_mult"] else 0.0
    return (row["corpus"], row["image"], int(row["K"]), mult)


def main() -> None:
    corpora = [d for d in corpus_dirs(CORPORA) if d.name == "kodak"]
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
        raise SystemExit("no .png/.jpg/.ppm images found in any corpus")

    prov = provenance(all_images, HERE.parent.parent)

    n_dithers = len(all_images) * len(KS) * (1 + len(EPS_MULTS))
    print(f"{len(all_images)} images x {len(KS)} K x (1 baseline + "
          f"{len(EPS_MULTS)} eps) = {n_dithers} dithers", flush=True)

    is_new_file = not CSV_PATH.exists()
    fh = CSV_PATH.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if is_new_file:
        writer.writeheader()
        fh.flush()

    n_new = 0
    t0 = time.time()
    try:
        for corpus in corpora:
            images = images_in(corpus)
            for path in images:
                oklab = load_oklab(path)
                for K in KS:
                    palette, _ = extract_palette(oklab, n_colours=K, seed=SEED)
                    if (path.name, K, "") not in done:
                        dithered, _ = diffuse(oklab, palette, shaping=None)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "variant": "baseline",
                            "eps_mult": "", "eps": "",
                        }
                        row.update(score(dithered, oklab))
                        row.update(prov)
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                    for mult in EPS_MULTS:
                        if (path.name, K, str(mult)) in done:
                            continue
                        eps = DEFAULT_EPS * mult
                        M, _, _ = global_shaping_matrix(palette, eps=eps)
                        dithered, _ = diffuse(oklab, palette, shaping=M)
                        row = {
                            "corpus": corpus.name, "image": path.name,
                            "K": K, "variant": "global",
                            "eps_mult": mult, "eps": eps,
                        }
                        row.update(score(dithered, oklab))
                        row.update(prov)
                        writer.writerow(row)
                        fh.flush()
                        n_new += 1
                print(f"  {path.name}: done ({n_new} new, "
                      f"{time.time() - t0:.0f}s elapsed)", flush=True)
    finally:
        fh.close()

    finalise_csv(CSV_PATH, FIELDS, _row_key)
    print(f"\n{n_new} new trial(s) added; {n_new + len(done)} total in {CSV_PATH}", flush=True)


if __name__ == "__main__":
    main()
