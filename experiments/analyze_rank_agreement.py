r"""Check whether S-CIELAB dE and L-channel noise rank conditions the same way.

This is the first analysis to run on the results (see PREREGISTRATION.md).
It reads a dithering-quality CSV and reports, pooled over all rows (both
corpora):

- winner concordance: the fraction of (image, K) where the best condition by
  `noise_L` is also the best by `scielab_de`;
- Spearman rho between `noise_L` and `scielab_de` over all (image, K, variant)
  rows;
- Spearman rho between the two metrics' *shaping improvement*, computed from
  the `baseline` and `global` variants as (baseline - global) per (image, K).
  Groups where either variant is absent are skipped from the improvement
  statistic.

Agreement gives the planned paper (S-CIELAB carries the claim, Oklab metrics
are the mechanism). Disagreement changes the paper's structure, so know which
one you have before writing.

Run from the repository root::

    python experiments/analyze_rank_agreement.py \
        experiments/dithering_quality/results/dithering_quality.csv
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", type=Path,
                    default=Path("experiments/dithering_quality/results/dithering_quality.csv"),
                    nargs="?")
    args = ap.parse_args()

    if not args.csv.exists():
        raise SystemExit(f"{args.csv} not found. Re-run the experiment first.")

    with args.csv.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        raise SystemExit("empty CSV")
    if "scielab_de" not in rows[0]:
        raise SystemExit(
            f"{args.csv} has no scielab_de column; re-run the experiment to regenerate it."
        )

    var_col = "variant" if "variant" in rows[0] else "config"
    variants = sorted({r[var_col] for r in rows})

    # (image, K) -> {variant: {metric: value}}
    per_key = defaultdict(dict)
    for r in rows:
        key = (r["image"], int(r["K"]))
        per_key[key][r[var_col]] = {
            "noise_L": float(r["noise_L"]), "scielab_de": float(r["scielab_de"]),
        }

    wins_agree = 0
    total = 0
    pooled_noise, pooled_scielab = [], []
    imp_noise, imp_scielab = [], []
    for key, variants_dict in per_key.items():
        total += 1
        best_noise = min(variants_dict, key=lambda v: variants_dict[v]["noise_L"])
        best_scielab = min(variants_dict, key=lambda v: variants_dict[v]["scielab_de"])
        wins_agree += int(best_noise == best_scielab)
        for v, d in variants_dict.items():
            pooled_noise.append(d["noise_L"])
            pooled_scielab.append(d["scielab_de"])
        if "baseline" in variants_dict and "global" in variants_dict:
            imp_noise.append(variants_dict["baseline"]["noise_L"]
                             - variants_dict["global"]["noise_L"])
            imp_scielab.append(variants_dict["baseline"]["scielab_de"]
                               - variants_dict["global"]["scielab_de"])

    rho_pooled, _ = spearmanr(pooled_noise, pooled_scielab)
    rho_improvement, _ = spearmanr(imp_noise, imp_scielab)

    print(f"CSV: {args.csv}  ({len(rows)} rows, {total} image-K groups, "
          f"variants: {variants})")
    print(f"winner concordance (best by noise_L == best by scielab_de): "
          f"{wins_agree}/{total} = {wins_agree / total:.1%}")
    print(f"Spearman rho (pooled noise_L vs scielab_de): {rho_pooled:+.3f}")
    if imp_noise:
        print(f"Spearman rho (shaping improvement, noise_L vs scielab_de): "
              f"{rho_improvement:+.3f}")


if __name__ == "__main__":
    main()
