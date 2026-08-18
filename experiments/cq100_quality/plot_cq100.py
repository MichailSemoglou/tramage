"""Plot CQ100 dithering-quality results: metrics by K, and gain vs palette geometry.

Mirrors ``experiments/dithering_quality/plot_quality.py`` with CQ100-specific
paths and an additional per-corpus (i.e., per-category when the dataset is
organised into category subdirectories) breakdown.

Run from the repository root, after run_cq100.py::

    python experiments/cq100_quality/plot_cq100.py
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

from tramage._harness import QUALITY_NUMERIC_FIELDS, percentile, read_rows

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "cq100_dithering_quality.csv"
GEOMETRY_CSV = HERE.parent / "palette_geometry" / "results" / "palette_geometry.csv"

VARIANT_COLORS = {"baseline": "#888888", "global": "#2c3e60", "cell": "#b94a48"}
VARIANT_ORDER = ["baseline", "global", "cell"]
METRICS = [
    ("noise_L", "L-channel noise (std of error)", False),
    ("tone_error", "blurred tone error (Oklab dE)", False),
    ("lowfreq_frac", "low-frequency error fraction", False),
    ("scielab_de", "S-CIELAB ΔE (perceptual)", False),
]


def _load() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_cq100.py first.")
    return read_rows(CSV_PATH, QUALITY_NUMERIC_FIELDS)




def plot_metrics_vs_K(rows: list[dict]) -> None:
    Ks = sorted({int(row["K"]) for row in rows})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.6), sharey=False)
    for ax, (metric, label, _) in zip(axes, METRICS):
        for variant in VARIANT_ORDER:
            medians, p25s, p75s = [], [], []
            for K in Ks:
                vals = np.array([
                    row[metric] for row in rows
                    if int(row["K"]) == K and row["variant"] == variant
                ])
                medians.append(np.median(vals))
                p25s.append(percentile(vals, 25))
                p75s.append(percentile(vals, 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=VARIANT_COLORS[variant],
                    label=variant, linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=VARIANT_COLORS[variant], alpha=0.15)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="variant", fontsize=9)
    fig.suptitle("CQ100 dithering quality by K: baseline vs global vs per-cell shaping\n"
                 "(line = median across 100 images; band = 25-75 percentile)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'metrics_vs_K.png'}")


def plot_gain_vs_geometry(rows: list[dict]) -> None:
    if not GEOMETRY_CSV.exists():
        print(f"skipping gain_vs_geometry: {GEOMETRY_CSV} not found "
              f"(run palette_geometry/run_geometry.py first)")
        return

    degeneracy: dict[tuple[str, int], float] = {}
    with GEOMETRY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["seed"]) != 0:
                continue
            lam3 = max(float(row["lambda3"]), 1e-12)
            ratio = (float(row["lambda1"]) + float(row["lambda2"])) / lam3
            degeneracy[(row["image"], int(row["K"]))] = ratio

    pivoted: dict[tuple[str, int], dict[str, float]] = defaultdict(dict)
    for row in rows:
        key = (row["image"], int(row["K"]))
        pivoted[key][row["variant"]] = row["noise_L"]

    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    for variant, color in (("global", "#2c3e60"), ("cell", "#b94a48")):
        xs, ys = [], []
        for key, variants in pivoted.items():
            if variant not in variants or "baseline" not in variants:
                continue
            if key not in degeneracy:
                continue
            base = variants["baseline"]
            if base <= 0:
                continue
            improvement = 1.0 - variants[variant] / base
            xs.append(degeneracy[key])
            ys.append(improvement)
        ax.scatter(xs, ys, s=20, alpha=0.45, color=color, edgecolors="none",
                   label=f"{variant} shaping")

    ax.axhline(0.0, color="#999", linestyle="--", linewidth=1)
    ax.text(ax.get_xlim()[1] * 0.9, 0.02, "shaped quieter  ↑",
            fontsize=8, color="#555", ha="right")
    ax.text(ax.get_xlim()[1] * 0.9, -0.02, "↓  shaped noisier",
            fontsize=8, color="#555", ha="right", va="top")
    ax.set_xscale("log")
    ax.axvline(2.0, color="#bbb", linestyle=":", linewidth=1)
    ax.text(2.0, ax.get_ylim()[0], "  isotropic (2.0)", fontsize=7, color="#999", va="top")
    ax.set_xlabel("palette degeneracy score  (λ1+λ2)/λ3   [smaller = more degenerate]")
    ax.set_ylabel("relative noise reduction   1 − noise_L(shaped)/noise_L(baseline)")
    ax.set_title("CQ100: does shaping's benefit track how degenerate the palette is?",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "gain_vs_geometry.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'gain_vs_geometry.png'}")


def main() -> None:
    rows = _load()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")
    plot_metrics_vs_K(rows)
    plot_gain_vs_geometry(rows)


if __name__ == "__main__":
    main()
