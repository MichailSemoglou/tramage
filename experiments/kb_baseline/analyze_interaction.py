"""Statistical analysis of the shaping x filter interaction (kb_baseline).

The FS/KB pair is a 2x2 factorial. This script reports the interaction two
ways, with the trend test as the PRIMARY analysis:

- **Primary: trend test.** A **Jonckheere-Terpstra** ordered-trend test
  across K, which tests a monotonic decline without committing to a functional
  form (a linear-in-log₂(K) fit is a poor model of the plateauing interaction,
  so its slope and p-value would inherit that misspecification). Reports the JT
  statistic and its one-sided p; the mean per-image linear slope is reported as
  a descriptive effect size only.
- **Secondary: per-K TOST.** A two-one-sided-tests equivalence test per K
  against a pre-specified margin, theta = 0.25 x the pooled shaping benefit,
  with a 90% bootstrap CI. Used to locate the crossing point, not as six
  independent findings.

Absolute interaction is reported alongside the ratio throughout: the ratio is
only meaningful where the shaping effect (its denominator) is not near zero,
and at large K the effect collapses so the ratio inflates.

Run from the repository root, after run_kbbaseline.py::

    python experiments/kb_baseline/analyze_interaction.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy import stats

from tramage._harness import read_rows

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CSV_PATH = RESULTS / "kbbaseline.csv"

# Pre-specified equivalence margin as a fraction of the main shaping effect:
# the interaction is deemed "negligible" if its 90% CI lies within
# +/- 25% of the pooled shaping benefit at that K.
EQUIV_MARGIN_FRACTION = 0.25
ALPHA = 0.05          # one-sided; 90% CI for the TOST
CI = 0.90
N_RESAMPLES = 10_000
SEED = 0

METRICS = [
    ("noise_L", "L-channel noise"),
    ("tone_error", "blurred tone error"),
    ("scielab_de", "S-CIELAB dE"),
]


def _bootstrap_interaction(interaction, main, n_resamples=N_RESAMPLES, seed=SEED):
    """Bootstrap means of the interaction and main effect, paired at image level."""
    interaction = np.asarray(interaction, dtype=np.float64)
    main = np.asarray(main, dtype=np.float64)
    n = interaction.size
    rng = np.random.default_rng(seed)
    inter_means = np.empty(n_resamples)
    main_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, n, n)
        inter_means[i] = interaction[idx].mean()
        main_means[i] = main[idx].mean()
    return inter_means, main_means


def _pct(a, q):
    return float(np.percentile(a, q))


def _trend_test(per_image: dict, Ks: list[int]):
    """Jonckheere-Terpstra ordered-trend test of a monotonic decline in K.

    Treats each (image, K) interaction as one observation grouped by K and
    tests the alternative that the interaction decreases with K. No parametric
    shape is assumed. Returns ``(jt, z, p, mean_slope)``; ``mean_slope`` is the
    mean per-image linear-in-log2(K) slope, reported only as a descriptive
    effect size (no p-value, since that fit is a poor model of the plateauing
    interaction).
    """
    groups = []
    for K in Ks:
        vals = [per_image[img][K] for img in per_image if K in per_image[img]]
        groups.append(np.asarray(vals))
    # Decreasing trend: count pairs (x at earlier K, y at later K) with x > y.
    jt = 0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            for x in groups[i]:
                jt += int((x > groups[j]).sum())
    nk = np.array([g.size for g in groups])
    N = int(nk.sum())
    E = (N * N - int((nk * nk).sum())) / 4.0
    V = (N * N * (2 * N + 3) - int((nk * nk * (2 * nk + 3)).sum())) / 72.0
    z = (jt - E) / np.sqrt(V)
    p = float(stats.norm.sf(z))  # one-sided: decreasing trend
    logKs = np.log2(Ks)
    slopes = []
    for img, by_K in per_image.items():
        ys = [by_K[K] for K in Ks if K in by_K]
        if len(ys) >= 2:
            slopes.append(np.polyfit(logKs[: len(ys)], ys, 1)[0])
    mean_slope = float(np.mean(slopes)) if slopes else float("nan")
    return int(jt), float(z), p, mean_slope


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit(f"{CSV_PATH} not found. Run run_kbbaseline.py first.")
    with CSV_PATH.open(newline="") as fh:
        header = next(csv.DictReader(fh)).keys()
    available = [m for m, _ in METRICS if m in header]
    rows = read_rows(CSV_PATH, ("K", *available))

    by_key = {(r["image"], int(r["K"]), r["config"]): r for r in rows}
    images = sorted({r["image"] for r in rows})
    Ks = sorted({int(r["K"]) for r in rows})

    out_rows = []
    for metric in available:
        print(f"\n=== {metric} ===")
        inter_matrix = {img: {} for img in images}
        print(f"{'K':>4} {'n':>3} {'main':>9} {'interact':>9} {'90% CI':>21} "
              f"{'ratio':>7} {'ratio CI':>21} {'theta':>7} {'p_up':>6} {'p_lo':>6} {'equiv':>6}")
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
            fs_benefit = np.asarray(fs_benefit)
            kb_benefit = np.asarray(kb_benefit)
            interaction = fs_benefit - kb_benefit
            main_effect = (fs_benefit + kb_benefit) / 2.0
            n = interaction.size
            for j, image in enumerate(images):
                inter_matrix[image][K] = float(interaction[j])

            inter_means, main_means = _bootstrap_interaction(interaction, main_effect)
            inter_mean = float(interaction.mean())
            inter_lo = _pct(inter_means, 100 * (1 - CI) / 2)
            inter_hi = _pct(inter_means, 100 - 100 * (1 - CI) / 2)
            main_mean = float(main_effect.mean())
            theta = EQUIV_MARGIN_FRACTION * main_mean

            # ratio of interaction to main effect (scale-free), with CI
            ratio = inter_mean / main_mean if abs(main_mean) > 1e-12 else float("nan")
            ratio_means = inter_means / main_means
            ratio_lo = _pct(ratio_means, 100 * (1 - CI) / 2)
            ratio_hi = _pct(ratio_means, 100 - 100 * (1 - CI) / 2)

            # TOST: two one-sided tests against +/- theta
            p_upper = float((inter_means <= -theta).mean())  # H0: mean <= -theta
            p_lower = float((inter_means >= +theta).mean())  # H0: mean >= +theta
            equiv = p_upper < ALPHA and p_lower < ALPHA

            out_rows.append({
                "metric": metric, "K": K, "n_images": n,
                "main_effect": main_mean,
                "interaction": inter_mean,
                "interaction_lo": inter_lo, "interaction_hi": inter_hi,
                "ratio": ratio, "ratio_lo": ratio_lo, "ratio_hi": ratio_hi,
                "theta": theta,
                "tost_p_upper": p_upper, "tost_p_lower": p_lower,
                "equivalent": equiv,
            })
            print(f"{K:>4} {n:>3} {main_mean:>9.4f} {inter_mean:>+9.4f} "
                  f"[{inter_lo:>+7.4f},{inter_hi:>+7.4f}] "
                  f"{ratio:>+7.2f} [{ratio_lo:>+7.2f},{ratio_hi:>+7.2f}] "
                  f"{theta:>7.4f} {p_upper:>6.3f} {p_lower:>6.3f} {'Y' if equiv else 'N':>6}")

        jt, z, p, mean_slope = _trend_test(inter_matrix, Ks)
        print(f"  PRIMARY trend test (Jonckheere-Terpstra): JT = {jt}  z = {z:+.2f}  "
              f"one-sided p = {p:.3g} (decreasing in K); "
              f"descriptive slope = {mean_slope:+.4f}/doubling")

    out_path = RESULTS / "interaction_statistics.csv"
    with out_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out_rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    print(f"\nwrote {out_path}")
    print(f"equivalence margin: theta = {EQUIV_MARGIN_FRACTION:.0%} x main shaping effect; "
          f"alpha = {ALPHA} (one-sided); bootstrap CI = {CI:.0%}")


if __name__ == "__main__":
    main()
