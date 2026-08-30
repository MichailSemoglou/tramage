"""Robustness of the KB two-regime observation (exploratory, post-run).

The observation: the Kolpatzik-Bouman filter's low-frequency excess over
Floyd-Steinberg (``kb-baseline - fs-baseline`` in ``lowfreq_frac``) changes
sign around a baseline low-frequency energy fraction of ~0.02 — below it KB
sits on a DC-leak floor, above it the HVS weighting wins. This split was
observed after the re-run (it is not in `PREREGISTRATION.md`), and the 0.02
boundary was chosen by eye, so this script reports the finding the honest
way:

1. **Threshold robustness** — the same medians at thresholds 0.01, 0.015,
   0.02, and 0.03, over the figure's K window (16-64) and over all K.
2. **Per-K split** at the 0.02 threshold, because baseline lowfreq_frac is
   strongly K-dependent: the below/above split partly re-describes K.
3. **The perceptual caveat** — the share of cells where KB beats FS on
   S-CIELAB ΔE per regime. The regime finding is a low-frequency-energy
   statement; on the primary endpoint KB does not win the above-boundary
   regime.

Also reconciles the two published cuts: ROADMAP section 8 (19 Aug) quotes
+0.0053 / -0.0109 and Figure 1 shows +0.005 / -0.006; the difference is the
K window (all K vs 16-64), not a discrepancy — both are printed here.

Run from the repository root, after run_kbbaseline.py::

    python experiments/kb_baseline/analyze_regimes.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "kbbaseline.csv"
OUT_CSV = RESULTS / "regime_robustness.csv"

KS = (8, 16, 32, 64, 128, 256)
THRESHOLDS = (0.01, 0.015, 0.02, 0.03)


def median(values: list[float]) -> float:
    v = sorted(values)
    n = len(v)
    return v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2])


def regime_stats(
    cells: list[tuple[float, float, float, float]],
    threshold: float,
) -> tuple[list[float], list[float], float, float]:
    """Split cells on baseline lowfreq; return (excess below, excess above,
    KB S-CIELAB win rate below, win rate above)."""
    lo = [(lf, ex, fs_de, kb_de) for lf, ex, fs_de, kb_de in cells if lf < threshold]
    hi = [(lf, ex, fs_de, kb_de) for lf, ex, fs_de, kb_de in cells if lf >= threshold]

    def win_rate(part: list[tuple[float, float, float, float]]) -> float:
        if not part:
            return float("nan")
        return sum(1 for _, _, fs_de, kb_de in part if kb_de < fs_de) / len(part)

    return (
        [ex for _, ex, _, _ in lo],
        [ex for _, ex, _, _ in hi],
        win_rate(lo),
        win_rate(hi),
    )


def main() -> None:
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))

    # (image, K) -> per-config (lowfreq_frac, scielab_de)
    by_cell: dict[tuple[str, int], dict[str, tuple[float, float]]] = defaultdict(dict)
    for r in rows:
        by_cell[(r["image"], int(r["K"]))][r["config"]] = (
            float(r["lowfreq_frac"]),
            float(r["scielab_de"]),
        )

    # cell = (fs baseline lowfreq, KB-FS lowfreq excess, fs de, kb de)
    cells: list[tuple[str, int, float, float, float, float]] = []
    for (image, K), d in by_cell.items():
        if "fs-baseline" in d and "kb-baseline" in d:
            fs_lf, fs_de = d["fs-baseline"]
            kb_lf, kb_de = d["kb-baseline"]
            cells.append((image, K, fs_lf, kb_lf - fs_lf, fs_de, kb_de))

    out_rows = []
    print("== 1. Threshold robustness ==")
    for window_name, window_ks in (("K=16-64", (16, 32, 64)), ("all K", KS)):
        window = [(lf, ex, fs_de, kb_de) for _, K, lf, ex, fs_de, kb_de in cells
                  if K in window_ks]
        print(f"  [{window_name}, n={len(window)}]")
        for thr in THRESHOLDS:
            lo, hi, win_lo, win_hi = regime_stats(window, thr)
            out_rows.append({
                "window": window_name, "threshold": thr,
                "n_below": len(lo), "n_above": len(hi),
                "median_excess_below": f"{median(lo):+.5f}" if lo else "",
                "median_excess_above": f"{median(hi):+.5f}" if hi else "",
                "scielab_kb_winrate_below": f"{win_lo:.3f}",
                "scielab_kb_winrate_above": f"{win_hi:.3f}",
            })
            print(f"    thr={thr:<6} below n={len(lo):>3} med {median(lo):+.5f} "
                  f"(KB S-CIELAB wins {win_lo:.0%}) | above n={len(hi):>3} "
                  f"med {median(hi):+.5f} (wins {win_hi:.0%})")

    print("\n== 2. Per-K split at threshold 0.02 ==")
    for K in KS:
        window = [(lf, ex, fs_de, kb_de) for _, k, lf, ex, fs_de, kb_de in cells if k == K]
        lo, hi, win_lo, win_hi = regime_stats(window, 0.02)
        share_lo = len(lo) / len(window)
        out_rows.append({
            "window": f"K={K}", "threshold": 0.02,
            "n_below": len(lo), "n_above": len(hi),
            "median_excess_below": f"{median(lo):+.5f}" if lo else "",
            "median_excess_above": f"{median(hi):+.5f}" if hi else "",
            "scielab_kb_winrate_below": f"{win_lo:.3f}",
            "scielab_kb_winrate_above": f"{win_hi:.3f}",
        })
        print(f"    K={K:>3}: {share_lo:.0%} below | below med "
              f"{median(lo):+.5f} (wins {win_lo:.0%}) | above med "
              f"{median(hi):+.5f} (wins {win_hi:.0%})"
              if lo and hi else f"    K={K:>3}: {share_lo:.0%} below (one regime empty)")

    with OUT_CSV.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f"\nwrote {OUT_CSV}")


if __name__ == "__main__":
    main()
