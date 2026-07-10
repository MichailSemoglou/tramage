"""Plot the Oklab geometry of extracted palettes from the results CSV.

Two figures, each designed to answer the reframing question directly:

- ``by_K.png`` -- stacked bars showing the distribution of palette
  degeneracy classes per K value. The ">=1 flat axis" share is the single
  most decision-relevant number: if it is high across K, the broad claim
  that extracted palettes are consistently near-degenerate holds; if it
  collapses past some K, the claim narrows to small-palette regimes.
- ``eigenvalue_ratio.png`` -- a continuous degeneracy score per palette
  (``(lambda1+lambda2)/lambda3``), shown as a strip per K. Complements
  the discrete classification by showing the spread, not just the modes.

Uses the standard-library ``csv`` module rather than pandas, to keep this
experiment's dependency surface at matplotlib (already used by the gallery)
plus the package itself.

Run from the repository root, after run_geometry.py::

    python experiments/palette_geometry/plot_geometry.py
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
CSV_PATH = HERE / "results" / "palette_geometry.csv"
OUT = HERE / "results"


def _load_rows() -> list[dict]:
    if not CSV_PATH.exists():
        raise SystemExit(
            f"{CSV_PATH} not found. Run run_geometry.py first."
        )
    with CSV_PATH.open(newline="") as fh:
        rows = list(csv.DictReader(fh))
    # Coerce numeric columns once, here, rather than at every use site.
    numeric = ["K", "seed", "lambda1", "lambda2", "lambda3", "g1", "g2", "g3",
               "n_flat", "n_partial", "n_real"]
    for row in rows:
        for k in numeric:
            row[k] = float(row[k])
    return rows


def plot_degeneracy_by_K(rows: list[dict]) -> None:
    """Stacked bars: for each K, the fraction of palettes with 0/1/2/3 flat axes."""
    Ks = sorted({row["K"] for row in rows})
    by_K: dict[float, list[int]] = defaultdict(list)
    for row in rows:
        by_K[row["K"]].append(int(row["n_flat"]))

    buckets = [0, 1, 2, 3]
    fractions = np.zeros((len(buckets), len(Ks)))
    for j, K in enumerate(Ks):
        flats = by_K[K]
        counts = np.array([flats.count(b) for b in buckets], dtype=float)
        fractions[:, j] = counts / counts.sum()

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    bottom = np.zeros(len(Ks))
    colors = ["#2c3e60", "#5b7aa8", "#a9bdd6", "#e8eef5"]
    labels = [
        "0 flat axes (broad spread)",
        "1 flat axis",
        "2 flat axes (near-planar)",
        "3 flat axes (point-like)",
    ]
    for i, b in enumerate(buckets):
        ax.bar([str(int(K)) for K in Ks], fractions[i], bottom=bottom,
               color=colors[i], label=labels[i], edgecolor="white", linewidth=0.5)
        bottom += fractions[i]

    # Annotate the ">=1 flat axis" share: the decision-relevant aggregate.
    # Marked in red where the majority of palettes have >=1 flat axis (the
    # signal that the broad near-degeneracy claim holds at that K).
    ge1 = fractions[1:].sum(axis=0)
    for j, K in enumerate(Ks):
        ax.text(j, 1.02, f"{ge1[j]*100:.0f}%", ha="center", va="bottom",
                fontsize=10, fontweight="bold",
                color=("#b94a48" if ge1[j] >= 0.5 else "#7a7a7a"))

    ax.set_xlabel("palette size K")
    ax.set_ylabel("fraction of palettes")
    ax.set_ylim(0, 1.18)
    ax.set_title("Oklab degeneracy of k-means palettes, by palette size\n"
                 "(label above each bar = share with >=1 flat axis, red = majority)",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)
    fig.tight_layout()
    fig.savefig(OUT / "by_K.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT / 'by_K.png'}")


def plot_eigenvalue_ratio(rows: list[dict]) -> None:
    """Strip plot of the continuous degeneracy score (lambda1+lambda2)/lambda3 vs K.

    A perfectly isotropic palette has ratio 2 (three equal eigenvalues).
    A perfectly planar palette has lambda3 = 0, hence ratio = infinity; we
    clip to a sensible ceiling. Lower ratios mean more isotropic; higher
    mean more degenerate.
    """
    Ks = sorted({row["K"] for row in rows})
    by_K: dict[float, list[float]] = defaultdict(list)
    for row in rows:
        lam3 = max(row["lambda3"], 1e-12)
        ratio = (row["lambda1"] + row["lambda2"]) / lam3
        by_K[row["K"]].append(min(ratio, 50.0))  # ceiling; genuine planar pile up here

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    rng = np.random.default_rng(0)
    for j, K in enumerate(Ks):
        ratios = np.array(by_K[K])
        x = j + 0.2 * rng.uniform(-1, 1, size=len(ratios))
        ax.scatter(x, ratios, s=12, alpha=0.35, color="#2c3e60", edgecolors="none")
        # Median line per K, robust to the long tail.
        ax.plot([j - 0.25, j + 0.25], [np.median(ratios)] * 2,
                color="#b94a48", linewidth=2)

    ax.axhline(2.0, color="#999", linestyle="--", linewidth=1)
    ax.text(len(Ks) - 0.5, 2.05, "isotropic (ratio = 2)", fontsize=8, color="#777",
            ha="right")
    ax.set_xticks(range(len(Ks)))
    ax.set_xticklabels([str(int(K)) for K in Ks])
    ax.set_xlabel("palette size K")
    ax.set_ylabel("degeneracy ratio  (lambda1+lambda2) / lambda3")
    ax.set_yscale("log")
    ax.set_title("Per-palette degeneracy score by K (red = median)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "eigenvalue_ratio.png", dpi=130)
    plt.close(fig)
    print(f"wrote {OUT / 'eigenvalue_ratio.png'}")


def main() -> None:
    rows = _load_rows()
    print(f"loaded {len(rows)} rows from {CSV_PATH}")
    plot_degeneracy_by_K(rows)
    plot_eigenvalue_ratio(rows)


if __name__ == "__main__":
    main()
