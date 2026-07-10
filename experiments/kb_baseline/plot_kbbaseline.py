"""Plot the KB baseline experiment results.

One headline figure with two views:

- ``metrics_vs_K.png`` -- two panels (noise_L, tone_error), four lines
  per panel (fs-baseline, fs-shaped, kb-baseline, kb-shaped), with 25-75%
  shaded bands across images. At a glance: does kb-shaped sit below
  kb-baseline?
- ``relative_improvement_vs_K.png`` -- the cleaner comparison. For each
  metric, two lines showing the relative improvement from shaping under
  each filter: (fs-base - fs-shaped)/fs-base and (kb-base - kb-shaped)/
  kb-base. Answers: does shaping's benefit survive against the stronger
  filter?

Run from the repository root, after run_kbbaseline.py::

    python experiments/kb_baseline/plot_kbbaseline.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "kbbaseline.csv"

CONFIG_COLORS = {
    "fs-baseline": "#bbbbbb", "fs-shaped": "#2c3e60",
    "kb-baseline": "#e0a8a8", "kb-shaped": "#b94a48",
}
CONFIG_ORDER = ["fs-baseline", "fs-shaped", "kb-baseline", "kb-shaped"]
METRICS = [("noise_L", "L-channel noise"), ("tone_error", "blurred tone error")]


def _load() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_kbbaseline.py first.")
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for k in ("K", "eps", "bias_norm", "bias_L", "bias_a", "bias_b",
                  "noise_L", "noise_a", "noise_b", "tone_error",
                  "lowfreq_frac", "anisotropy"):
            row[k] = float(row[k])
    return rows


def _percentile(arr: np.ndarray, p: float) -> float:
    return float(np.percentile(arr, p)) if len(arr) else float("nan")


def plot_metrics_vs_K(rows: list[dict]) -> None:
    Ks = sorted({int(r["K"]) for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.8))
    for ax, (metric, label) in zip(axes, METRICS):
        for config in CONFIG_ORDER:
            medians, p25s, p75s = [], [], []
            for K in Ks:
                vals = np.array([r[metric] for r in rows
                                 if int(r["K"]) == K and r["config"] == config])
                medians.append(np.median(vals) if len(vals) else float("nan"))
                p25s.append(_percentile(vals, 25))
                p75s.append(_percentile(vals, 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=CONFIG_COLORS[config],
                    label=config, linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=CONFIG_COLORS[config], alpha=0.13)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="config", fontsize=8, loc="upper right")
    fig.suptitle("Kodak dithering quality: Floyd-Steinberg vs Kolpatzik-Bouman baseline\n"
                 "(line = median across 24 images; band = 25-75 percentile)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'metrics_vs_K.png'}")


def plot_relative_improvement(rows: list[dict]) -> None:
    """Relative shaping improvement, separately under each filter."""
    Ks = sorted({int(r["K"]) for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.8))
    for ax, (metric, label) in zip(axes, METRICS):
        for filter_name, base_c, shaped_c, color in (
            ("FS", "fs-baseline", "fs-shaped", "#2c3e60"),
            ("KB", "kb-baseline", "kb-shaped", "#b94a48"),
        ):
            medians, p25s, p75s = [], [], []
            for K in Ks:
                improvements = []
                for image in sorted({r["image"] for r in rows}):
                    base = next((r[metric] for r in rows if r["image"] == image
                                 and int(r["K"]) == K and r["config"] == base_c), None)
                    shaped = next((r[metric] for r in rows if r["image"] == image
                                   and int(r["K"]) == K and r["config"] == shaped_c), None)
                    if base is None or shaped is None or base <= 0:
                        continue
                    improvements.append((base - shaped) / base)
                medians.append(np.median(improvements) if improvements else float("nan"))
                p25s.append(_percentile(np.array(improvements), 25))
                p75s.append(_percentile(np.array(improvements), 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=color,
                    label=f"shaping under {filter_name}", linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=color, alpha=0.15)
        ax.axhline(0.0, color="#999", linestyle="--", linewidth=1)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(f"relative improvement\n({label})")
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="filter", fontsize=9, loc="upper right")
    fig.suptitle("Does shaping's benefit survive against the stronger KB filter?\n"
                 "Relative improvement over each filter's own baseline "
                 "(line = median, band = 25-75%)", fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "relative_improvement_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'relative_improvement_vs_K.png'}")


def main() -> None:
    rows = _load()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")
    plot_metrics_vs_K(rows)
    plot_relative_improvement(rows)


if __name__ == "__main__":
    main()
