"""Plot the Oklab-vs-CIELAB head-to-head results.

Two figures, both summarizing the full 124-image corpus (Kodak24 + CQ100;
the CSV of record defines the scope):

- ``relative_improvement_vs_K.png`` -- the headline plot. For each metric,
  two lines showing the relative shaping improvement vs K: one for Oklab
  shaping, one for CIELAB shaping. Baseline is the implicit zero line
  (relative improvement = 0). Unit-free, so the two spaces are directly
  comparable here: whichever line sits higher is the space where shaping
  has more purchase.
- ``raw_metrics_vs_K.png`` -- raw metric values per K, four lines
  (oklab-baseline/oklab/cielab-baseline/cielab). Raw values are NOT
  cross-comparable between spaces (CIELAB L* is 0-100, Oklab L is ~0-1),
  so the lines are for transparency and shape comparison only, not for
  reading off "which is lower". A note on the figure says so.

Run from the repository root, after run_headtohead.py::

    python experiments/colour_space/plot_headtohead.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from tramage._harness import QUALITY_NUMERIC_FIELDS, percentile, read_rows

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "headtohead.csv"

CONFIG_COLORS = {
    "oklab-baseline": "#888888", "oklab": "#2c3e60",
    "cielab-baseline": "#e0a8a8", "cielab": "#b94a48",
}
CONFIG_ORDER = ["oklab-baseline", "oklab", "cielab-baseline", "cielab"]

# Metrics and how to interpret "improvement". For all of these, lower is
# better, so "improvement" is positive when shaped < baseline.
METRICS = [
    ("noise_L", "L-channel noise"),
    ("tone_error", "blurred tone error"),
    ("lowfreq_frac", "low-frequency error fraction"),
    ("scielab_de", "S-CIELAB ΔE (on RGB, cross-comparable)"),
]


def _load() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_headtohead.py first.")
    return read_rows(CSV_PATH, QUALITY_NUMERIC_FIELDS)




def plot_relative_improvement(rows: list[dict]) -> None:
    """Headline: relative improvement (baseline - shaped) / baseline, per space.

    Each space's shaping is compared against its OWN baseline (oklab vs
    oklab-baseline, cielab vs cielab-baseline), so the comparison is
    coherent within each space rather than mixing Oklab L (~0-1) with
    CIELAB L* (0-100).
    """
    Ks = sorted({int(r["K"]) for r in rows})
    by_key = {(r["image"], int(r["K"]), r["config"]): r for r in rows}
    images = sorted({r["image"] for r in rows})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.6))
    for ax, (metric, label) in zip(axes, METRICS):
        for space, base_c, color in (
            ("Oklab", "oklab-baseline", "#2c3e60"),
            ("CIELAB", "cielab-baseline", "#b94a48"),
        ):
            medians, p25s, p75s = [], [], []
            for K in Ks:
                improvements = []
                for image in images:
                    base_row = by_key.get((image, K, base_c))
                    shaped_row = by_key.get((image, K, space.lower()))
                    if base_row is None or shaped_row is None:
                        continue
                    base, shaped = base_row[metric], shaped_row[metric]
                    if base <= 0:
                        continue
                    improvements.append((base - shaped) / base)
                medians.append(np.median(improvements) if improvements else float("nan"))
                p25s.append(percentile(np.array(improvements), 25))
                p75s.append(percentile(np.array(improvements), 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=color, label=f"{space} shaping", linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=color, alpha=0.15)
        ax.axhline(0.0, color="#999", linestyle="--", linewidth=1)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(f"relative improvement\n({label})")
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="shaping space", fontsize=9, loc="upper right")
    fig.suptitle("Which color space gives palette-structure shaping more purchase?\n"
                 "Relative improvement over each space's own baseline "
                 "(line = median, band = 25-75%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "relative_improvement_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'relative_improvement_vs_K.png'}")


def plot_raw_metrics(rows: list[dict]) -> None:
    """Transparency: raw metric values. Not cross-comparable between spaces."""
    Ks = sorted({int(r["K"]) for r in rows})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.6))
    for ax, (metric, label) in zip(axes, METRICS):
        for config in CONFIG_ORDER:
            medians = []
            for K in Ks:
                vals = [r[metric] for r in rows
                        if int(r["K"]) == K and r["config"] == config]
                medians.append(np.median(vals) if vals else float("nan"))
            ax.plot([str(K) for K in Ks], medians, marker="o",
                    color=CONFIG_COLORS[config], label=config, linewidth=1.8)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="config", fontsize=9)
    fig.suptitle("Raw metrics by K (NOT cross-comparable between Oklab and CIELAB -- "
                 "different units; see relative_improvement plot for the fair comparison)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "raw_metrics_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'raw_metrics_vs_K.png'}")


def main() -> None:
    rows = _load()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")
    plot_relative_improvement(rows)
    plot_raw_metrics(rows)


if __name__ == "__main__":
    main()
