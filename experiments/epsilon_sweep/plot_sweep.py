"""Plot the epsilon-sweep CSV as a noise-vs-tone Pareto front and ΔE curve.

Produces two figures:

1. ``results/sweep_pareto.png``: noise_std vs blurred_tone_error, with
   eps color-coded along the curve.
2. ``results/sweep_scielab.png``: S-CIELAB ΔE vs eps, with the default
   eps marked.

Run after ``run_sweep.py``::

    python experiments/epsilon_sweep/plot_sweep.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

from tramage.shaping import DEFAULT_EPS  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"


def _load_csv() -> list[dict]:
    rows: list[dict] = []
    with (RESULTS / "epsilon_sweep.csv").open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return rows


def _pareto_figure(rows: list[dict]) -> None:
    epsilons = np.array([float(r["eps"]) for r in rows])
    noise = np.array([float(r["noise_L"]) for r in rows])
    tone = np.array([float(r["tone_error"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    scatter = ax.scatter(noise, tone, c=np.log10(epsilons), cmap="viridis_r",
                         edgecolors="k", linewidths=0.3, s=60, zorder=3)
    ax.plot(noise, tone, color="0.5", linewidth=0.8, zorder=2)

    idx = int(np.argmin(np.abs(epsilons - DEFAULT_EPS)))  # nearest grid point
    ax.scatter(noise[idx], tone[idx], marker="*", s=200,
               color="tab:red", edgecolors="k", linewidths=0.5,
               zorder=4, label=f"default eps ({DEFAULT_EPS:.2e})")

    fig.colorbar(scatter, ax=ax, label="log10(eps)")
    ax.set_xlabel("noise std (L channel)")
    ax.set_ylabel("blurred tone error")
    ax.set_title("Epsilon sweep: noise vs tone Pareto front")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "sweep_pareto.png", dpi=150)
    plt.close(fig)


def _scielab_figure(rows: list[dict]) -> None:
    epsilons = np.array([float(r["eps"]) for r in rows])
    de = np.array([float(r["scielab_de"]) for r in rows])

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogx(epsilons, de, "o-", markersize=5, linewidth=1.2, color="tab:blue")
    ax.axvline(DEFAULT_EPS, color="tab:red", linestyle="--", linewidth=1,
               label=f"default eps ({DEFAULT_EPS:.2e})")
    ax.set_xlabel("eps")
    ax.set_ylabel("S-CIELAB ΔE")
    ax.set_title("S-CIELAB ΔE vs epsilon")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(RESULTS / "sweep_scielab.png", dpi=150)
    plt.close(fig)


def main() -> None:
    path = RESULTS / "epsilon_sweep.csv"
    if not path.exists():
        raise SystemExit(f"{path} not found. Run run_sweep.py first.")
    rows = _load_csv()
    print(f"loaded {len(rows)} rows from {path}")
    _pareto_figure(rows)
    print(f"wrote {RESULTS / 'sweep_pareto.png'}")
    _scielab_figure(rows)
    print(f"wrote {RESULTS / 'sweep_scielab.png'}")


if __name__ == "__main__":
    main()
