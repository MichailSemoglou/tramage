"""Pre-specified subgroup statistics for the S-CIELAB shaping benefit.

`PREREGISTRATION.md` declares Kodak24 and CQ100 as pre-specified subgroups
of the 124-image corpus. The pooled headline (median per-image reduction,
sign test) can mask subgroup behavior: at K=256 the pooled null hides
uniform Kodak harm against a CQ100 gain, so this script reports, per K and
per group (pooled / kodak / cq100):

- n images, median and mean per-image S-CIELAB ΔE reduction (%);
- wins (shaped < baseline) with the two-sided sign-test p-value (the test
  the abstract quotes at K=256);
- the two-sided Wilcoxon signed-rank p-value on the absolute paired
  differences (baseline - shaped, in ΔE units), as a complement that uses
  magnitudes, not just signs.

Conventions: percent reductions are per-image (baseline - shaped)/baseline;
the sign test is two-sided binomial at p=0.5; Wilcoxon uses
``scipy.stats.wilcoxon(..., alternative="two-sided")`` on absolute ΔE
differences. Both p-values are descriptive companions to the effect sizes,
per the preregistration's reporting conventions.

Run from the repository root, after run_quality.py::

    python experiments/dithering_quality/analyze_subgroups.py
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DEFAULT_CSV = RESULTS / "dithering_quality.csv"
OUT_CSV = RESULTS / "subgroup_scielab.csv"

FIELDS = [
    "K", "group", "n", "median_pct", "mean_pct",
    "wins", "sign_p", "wilcoxon_p",
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    args = parser.parse_args()

    with open(args.csv, newline="") as fh:
        rows = list(csv.DictReader(fh))

    # (K, corpus) -> list of (baseline, shaped) S-CIELAB pairs, global variant
    pairs: dict[tuple[int, str], list[tuple[float, float]]] = defaultdict(list)
    by_cell: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for r in rows:
        by_cell[(r["image"], int(r["K"]))][r["variant"]] = float(r["scielab_de"])
    corpus_of = {r["image"]: r["corpus"] for r in rows}
    for (image, K), d in by_cell.items():
        if "baseline" in d and "global" in d:
            pairs[(K, corpus_of[image])].append((d["baseline"], d["global"]))

    out_rows = []
    for K in sorted({k for k, _ in pairs}):
        per_corpus = {c: pairs.get((K, c), []) for c in ("kodak", "cq100")}
        groups = {"pooled": per_corpus["kodak"] + per_corpus["cq100"], **per_corpus}
        for group, vals in groups.items():
            if not vals:
                continue
            n = len(vals)
            pcts = [100.0 * (b - s) / b for b, s in vals]
            diffs = [b - s for b, s in vals]
            wins = sum(1 for b, s in vals if s < b)
            sign_p = float(stats.binomtest(wins, n, 0.5).pvalue)
            wilcoxon_p = float(stats.wilcoxon(diffs, alternative="two-sided").pvalue)
            pcts_sorted = sorted(pcts)
            med = (pcts_sorted[n // 2] if n % 2
                   else 0.5 * (pcts_sorted[n // 2 - 1] + pcts_sorted[n // 2]))
            out_rows.append({
                "K": K, "group": group, "n": n,
                "median_pct": f"{med:.3f}", "mean_pct": f"{sum(pcts)/n:.3f}",
                "wins": wins, "sign_p": f"{sign_p:.4g}",
                "wilcoxon_p": f"{wilcoxon_p:.4g}",
            })

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(out_rows)

    header = (f"{'K':>4} {'group':<7} {'n':>4} {'med%':>7} {'mean%':>7} "
              f"{'wins':>9} {'sign_p':>9} {'wilx_p':>9}")
    print(header)
    for r in out_rows:
        print(f"{r['K']:>4} {r['group']:<7} {r['n']:>4} "
              f"{float(r['median_pct']):>7.2f} {float(r['mean_pct']):>7.2f} "
              f"{r['wins']:>4}/{r['n']:<4} {r['sign_p']:>9} {r['wilcoxon_p']:>9}")
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
