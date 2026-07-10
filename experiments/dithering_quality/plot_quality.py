"""Plot dithering-quality results: metrics by K, and gain vs palette geometry.

Two figures, each designed to land a specific claim:

- ``metrics_vs_K.png`` -- three panels (noise_L, tone_error, lowfreq_frac),
  three lines per panel (baseline/global/cell), with 25-75% shaded bands
  across images. The headline plot: at a glance, does shaped sit below
  baseline?
- ``gain_vs_geometry.png`` -- a scatter joining this CSV to the geometry
  CSV on (image, K). X = continuous degeneracy score
  ``(lambda1+lambda2)/lambda3``; Y = relative improvement
  ``1 - noise_L_shaped / noise_L_baseline``. Answers the mechanism
  question: does the magnitude of shaping's benefit track how degenerate
  the palette is?

Uses the standard-library ``csv`` module rather than pandas, to keep this
experiment's dependency surface at matplotlib plus the package itself.

Run from the repository root, after run_quality.py::

    python experiments/dithering_quality/plot_quality.py
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
QUALITY_CSV = RESULTS / "dithering_quality.csv"
GEOMETRY_CSV = HERE.parent / "palette_geometry" / "results" / "palette_geometry.csv"

VARIANT_COLORS = {"baseline": "#888888", "global": "#2c3e60", "cell": "#b94a48"}
VARIANT_ORDER = ["baseline", "global", "cell"]
METRICS = [
    ("noise_L", "L-channel noise (std of error)", False),  # lower is better
    ("tone_error", "blurred tone error (Oklab dE)", False),
    ("lowfreq_frac", "low-frequency error fraction", False),
]


def _load_quality() -> list[dict]:
    if not QUALITY_CSV.exists():
        raise SystemExit(f"{QUALITY_CSV} not found. Run run_quality.py first.")
    with QUALITY_CSV.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for k in ("K", "eps", "bias_norm", "bias_L", "bias_a", "bias_b",
                  "noise_L", "noise_a", "noise_b", "tone_error",
                  "lowfreq_frac", "anisotropy"):
            row[k] = float(row[k])
    return rows


def _percentile(arr: np.ndarray, p: float) -> float:
    if len(arr) == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def plot_metrics_vs_K(rows: list[dict]) -> None:
    Ks = sorted({int(row["K"]) for row in rows})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(14, 4.6), sharey=False)
    for ax, (metric, label, _) in zip(axes, METRICS):
        for variant in VARIANT_ORDER:
            # Per-K: collect this metric across all images for this variant.
            medians, p25s, p75s = [], [], []
            for K in Ks:
                vals = np.array([
                    row[metric] for row in rows
                    if int(row["K"]) == K and row["variant"] == variant
                ])
                medians.append(np.median(vals))
                p25s.append(_percentile(vals, 25))
                p75s.append(_percentile(vals, 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=VARIANT_COLORS[variant],
                    label=variant, linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=VARIANT_COLORS[variant], alpha=0.15)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="variant", fontsize=9)
    fig.suptitle("Kodak dithering quality by K: baseline vs global vs per-cell shaping\n"
                 "(line = median across 24 images; band = 25-75 percentile)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'metrics_vs_K.png'}")


def plot_gain_vs_geometry(rows: list[dict]) -> None:
    """Scatter: degeneracy score vs shaping's relative noise improvement."""
    if not GEOMETRY_CSV.exists():
        print(f"skipping gain_vs_geometry: {GEOMETRY_CSV} not found "
              f"(run palette_geometry/run_geometry.py first)")
        return

    # Build (image, K) -> degeneracy ratio from the geometry CSV (seed=0 only).
    degeneracy: dict[tuple[str, int], float] = {}
    with GEOMETRY_CSV.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if int(row["seed"]) != 0:
                continue
            lam3 = max(float(row["lambda3"]), 1e-12)
            ratio = (float(row["lambda1"]) + float(row["lambda2"])) / lam3
            degeneracy[(row["image"], int(row["K"]))] = ratio

    # Pivot quality rows: (image, K) -> {variant: noise_L}
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
            improvement = 1.0 - variants[variant] / base  # >0 means shaped is quieter
            # Log-x: degeneracy score spans several orders of magnitude.
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
    ax.set_title("Does shaping's benefit track how degenerate the palette is?",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(RESULTS / "gain_vs_geometry.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'gain_vs_geometry.png'}")


def main() -> None:
    rows = _load_quality()
    print(f"loaded {len(rows)} rows from {QUALITY_CSV}")
    plot_metrics_vs_K(rows)
    plot_gain_vs_geometry(rows)


if __name__ == "__main__":
    main()
