"""DV-filter analysis of the six-config kb_baseline run (S-CIELAB arbiter).

The six-config kb_baseline run adds dv-baseline and dv-shaped to the FS/KB
factorial. This script reproduces the four DV tables reported in
``experiments/kb_baseline/README.md`` (three quoted, one referenced) from
``results/kbbaseline.csv``,
on S-CIELAB ΔE, the one metric with spatial CSF weighting:

- **Table A**:median S-CIELAB ΔE per config per K (pooled, n = 124).
- **Table B**:shaping effect per filter: median per-image % ΔE reduction
  from baseline to shaped, with a two-sided Wilcoxon signed-rank p on the
  paired per-image ΔE values.
- **Table C**:the DV shaping effect split by corpus (Kodak n = 24,
  CQ100 n = 100): the pooled near-null masks a CQ100 gain and a Kodak harm.
- **Table D**:cross-filter comparisons: fs-shaped vs dv-baseline, and
  dv-baseline against the other two unshaped baselines.

The per-image % reduction of ``candidate`` relative to ``reference`` is
``100 * (reference - candidate) / reference``; positive means the candidate
has lower ΔE. The median is taken over images; the Wilcoxon test runs on
the paired ΔE values themselves, not on the percentages.

Run from the repository root, after run_kbbaseline.py::

    python experiments/kb_baseline/analyze_dv.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from tramage._harness import read_rows

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "kbbaseline.csv"

METRIC = "scielab_de"
CONFIGS = (
    "fs-baseline", "fs-shaped",
    "kb-baseline", "kb-shaped",
    "dv-baseline", "dv-shaped",
)
FILTERS = ("fs", "kb", "dv")
CORPORA = ("kodak", "cq100")


def _fmt_p(p: float) -> str:
    """Two significant figures; scientific notation below 0.001."""
    return f"{p:.1e}" if p < 1e-3 else f"{p:.2g}"


def _pct_reduction(by_key, images, K, reference, candidate):
    """Median per-image % ΔE reduction of ``candidate`` vs ``reference``.

    Positive means the candidate has lower ΔE. ``p`` is the two-sided
    Wilcoxon signed-rank p-value on the paired per-image ΔE values.
    """
    ref = np.array([by_key[(img, K, reference)] for img in images])
    cand = np.array([by_key[(img, K, candidate)] for img in images])
    median = float(np.median(100.0 * (ref - cand) / ref))
    p = float(stats.wilcoxon(ref, cand).pvalue)
    return median, p


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_kbbaseline.py first.")
    rows = read_rows(CSV_PATH, ("K", METRIC))
    by_key = {(r["image"], int(r["K"]), r["config"]): r[METRIC] for r in rows}
    corpus_of = {r["image"]: r["corpus"] for r in rows}
    images = sorted(corpus_of)
    Ks = sorted({int(r["K"]) for r in rows})

    print(f"\n=== Table A: median S-CIELAB dE per config per K "
          f"(pooled, n = {len(images)}) ===")
    short = ("fs-b", "fs-s", "kb-b", "kb-s", "dv-b", "dv-s")
    print(f"{'K':>4} " + " ".join(f"{s:>7}" for s in short))
    for K in Ks:
        cells = " ".join(
            f"{np.median([by_key[(i, K, c)] for i in images]):>7.3f}"
            for c in CONFIGS
        )
        print(f"{K:>4} {cells}")

    print(f"\n=== Table B: shaping effect per filter (median per-image % dE "
          f"reduction, baseline -> shaped; Wilcoxon p, n = {len(images)}) ===")
    print(f"{'K':>4} " + " ".join(f"{f:>18}" for f in FILTERS))
    for K in Ks:
        cells = []
        for f in FILTERS:
            med, p = _pct_reduction(by_key, images, K, f"{f}-baseline", f"{f}-shaped")
            cells.append(f"{med:+.1f}% ({_fmt_p(p)})")
        print(f"{K:>4} " + " ".join(f"{c:>18}" for c in cells))

    print("\n=== Table C: dv-shaped vs dv-baseline by corpus "
          "(median % reduction; Wilcoxon p) ===")
    labels = []
    for corpus in CORPORA:
        n = sum(1 for i in images if corpus_of[i] == corpus)
        labels.append(f"{corpus} (n={n})")
    print(f"{'K':>4} " + " ".join(f"{lab:>18}" for lab in labels))
    for K in Ks:
        cells = []
        for corpus in CORPORA:
            imgs = [i for i in images if corpus_of[i] == corpus]
            med, p = _pct_reduction(by_key, imgs, K, "dv-baseline", "dv-shaped")
            cells.append(f"{med:+.1f}% ({_fmt_p(p)})")
        print(f"{K:>4} " + " ".join(f"{c:>18}" for c in cells))

    # Each pair is (candidate, reference): the median % reduction of the
    # candidate relative to the reference.
    pairs = [
        ("fs-shaped", "dv-baseline", "fs-s vs dv-b"),
        ("dv-baseline", "fs-baseline", "dv-b vs fs-b"),
        ("dv-baseline", "kb-baseline", "dv-b vs kb-b"),
    ]
    print(f"\n=== Table D: cross-filter comparisons (median per-image % dE "
          f"reduction of the first config relative to the second, "
          f"positive = first better; Wilcoxon p, n = {len(images)}) ===")
    print(f"{'K':>4} " + " ".join(f"{label:>18}" for _, _, label in pairs))
    for K in Ks:
        cells = []
        for candidate, reference, _ in pairs:
            med, p = _pct_reduction(by_key, images, K, reference, candidate)
            cells.append(f"{med:+.1f}% ({_fmt_p(p)})")
        print(f"{K:>4} " + " ".join(f"{c:>18}" for c in cells))


if __name__ == "__main__":
    main()
