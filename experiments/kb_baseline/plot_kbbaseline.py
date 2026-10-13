"""Plot the filter-baseline experiment results.

Three figures:

- ``metrics_vs_K.png`` -- panels (noise_L, tone_error, S-CIELAB), six lines
  per panel (fs/kb/dv x baseline/shaped), 25-75% bands across images.
- ``relative_improvement_vs_K.png`` -- shaping's relative improvement under
  each filter, (base - shaped)/base, for the FS/KB factorial.
- ``interaction_vs_K.png`` -- the FS-KB interaction with a 90% bootstrap CI
  and the +/-25% equivalence-margin band (see analyze_interaction.py).

Run from the repository root, after run_kbbaseline.py::

    python experiments/kb_baseline/plot_kbbaseline.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

from tramage._harness import QUALITY_NUMERIC_FIELDS, bootstrap_ci, percentile, read_rows

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "kbbaseline.csv"

CONFIG_COLORS = {
    "fs-baseline": "#bbbbbb", "fs-shaped": "#2c3e60",
    "kb-baseline": "#e0a8a8", "kb-shaped": "#b94a48",
    "dv-baseline": "#c9d8c5", "dv-shaped": "#4a7c59",
}
CONFIG_ORDER = ["fs-baseline", "fs-shaped", "kb-baseline", "kb-shaped",
                "dv-baseline", "dv-shaped"]
METRICS = [
    ("noise_L", "L-channel noise"),
    ("tone_error", "blurred tone error"),
    ("scielab_de", "S-CIELAB ΔE (perceptual)"),
]


def _load() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_kbbaseline.py first.")
    return read_rows(CSV_PATH, QUALITY_NUMERIC_FIELDS)




def plot_metrics_vs_K(rows: list[dict]) -> None:
    """Metrics by K: median line per config with a 25-75% band across images."""
    Ks = sorted({int(r["K"]) for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.8))
    for ax, (metric, label) in zip(axes, METRICS):
        for config in CONFIG_ORDER:
            medians, p25s, p75s = [], [], []
            for K in Ks:
                vals = np.array([r[metric] for r in rows
                                 if int(r["K"]) == K and r["config"] == config])
                medians.append(np.median(vals) if len(vals) else float("nan"))
                p25s.append(percentile(vals, 25))
                p75s.append(percentile(vals, 75))
            xs = [str(K) for K in Ks]
            ax.plot(xs, medians, marker="o", color=CONFIG_COLORS[config],
                    label=config, linewidth=1.8)
            ax.fill_between(xs, p25s, p75s, color=CONFIG_COLORS[config], alpha=0.13)
        ax.set_xlabel("palette size K")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.3)
    axes[0].legend(title="config", fontsize=8, loc="upper right")
    fig.suptitle("Dithering quality by K: Floyd-Steinberg vs Kolpatzik-Bouman vs Damera-Venkata\n"
                 "(line = median across 124 images, Kodak24 + CQ100; band = 25-75 percentile)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(RESULTS / "metrics_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'metrics_vs_K.png'}")


def plot_relative_improvement(rows: list[dict]) -> None:
    """Relative shaping improvement, separately under each filter."""
    Ks = sorted({int(r["K"]) for r in rows})
    by_key = {(r["image"], int(r["K"]), r["config"]): r for r in rows}
    images = sorted({r["image"] for r in rows})
    fig, axes = plt.subplots(1, len(METRICS), figsize=(11, 4.8))
    for ax, (metric, label) in zip(axes, METRICS):
        for filter_name, base_c, shaped_c, color in (
            ("FS", "fs-baseline", "fs-shaped", "#2c3e60"),
            ("KB", "kb-baseline", "kb-shaped", "#b94a48"),
        ):
            medians, p25s, p75s = [], [], []
            for K in Ks:
                improvements = []
                for image in images:
                    base_row = by_key.get((image, K, base_c))
                    shaped_row = by_key.get((image, K, shaped_c))
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


def plot_interaction(rows: list[dict]) -> None:
    """Shaping x filter interaction, with a bootstrap CI and equivalence margin.

    Per K, the per-image shaping benefit is (baseline - shaped) under each
    filter; the interaction is their difference. The shaded band is the
    pre-specified equivalence margin +/- 25% of the pooled shaping benefit:
    where the CI lies inside the band, palette shaping is statistically
    indistinguishable from being independent of the filter.
    """
    Ks = sorted({int(r["K"]) for r in rows})
    by_key = {(r["image"], int(r["K"]), r["config"]): r for r in rows}
    images = sorted({r["image"] for r in rows})

    fig, axes = plt.subplots(1, len(METRICS), figsize=(len(METRICS) * 5.2, 4.8))
    for ax, (metric, label) in zip(axes, METRICS):
        means, los, his, thetas = [], [], [], []
        for K in Ks:
            fs_benefit, kb_benefit = [], []
            for image in images:
                fs_b = by_key.get((image, K, "fs-baseline"))
                fs_s = by_key.get((image, K, "fs-shaped"))
                kb_b = by_key.get((image, K, "kb-baseline"))
                kb_s = by_key.get((image, K, "kb-shaped"))
                if None in (fs_b, fs_s, kb_b, kb_s):
                    continue
                fs_benefit.append(fs_b[metric] - fs_s[metric])
                kb_benefit.append(kb_b[metric] - kb_s[metric])
            interaction = np.array(fs_benefit) - np.array(kb_benefit)
            main = (np.array(fs_benefit) + np.array(kb_benefit)) / 2.0
            lo, hi = bootstrap_ci(interaction, ci=0.90)
            means.append(float(interaction.mean()))
            los.append(lo)
            his.append(hi)
            thetas.append(0.25 * float(main.mean()))
        xs = np.arange(len(Ks))
        ax.axhline(0.0, color="#999", linestyle="--", linewidth=1)
        ax.fill_between(xs, -np.array(thetas), np.array(thetas),
                        color="#2c3e60", alpha=0.08)
        ax.errorbar(xs, means,
                    yerr=[np.array(means) - np.array(los), np.array(his) - np.array(means)],
                    fmt="o-", color="#2c3e60", linewidth=1.8, capsize=3)
        ax.set_xticks(xs)
        ax.set_xticklabels([str(K) for K in Ks])
        ax.set_xlabel("palette size K")
        ax.set_ylabel(f"interaction\n({label})\nFS benefit − KB benefit")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Shaping x filter interaction (mean ± 90% bootstrap CI)\n"
                 "shaded band = ±25% equivalence margin (theta = 0.25 x pooled "
                 "shaping benefit); CI inside band = orthogonal", fontsize=10)
    fig.tight_layout()
    fig.savefig(RESULTS / "interaction_vs_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {RESULTS / 'interaction_vs_K.png'}")


def main() -> None:
    rows = _load()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")
    plot_metrics_vs_K(rows)
    plot_relative_improvement(rows)
    plot_interaction(rows)


if __name__ == "__main__":
    main()
