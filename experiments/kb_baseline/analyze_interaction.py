"""Statistical analysis of the shaping x filter interaction (kb_baseline).

The FS/KB pair is a 2x2 factorial. This script reports the interaction two
ways, with the trend test as the PRIMARY analysis:

- **Primary — trend test.** A single-degree-of-freedom test: for each image,
  the slope of the interaction against log2(K); a one-sample t-test on the
  per-image slopes. This tests the monotonic-K claim in one step and sidesteps
  the multiplicity of six per-K tests. Reports the slope, 95% CI, p-value, and
  the K at which the fitted interaction crosses zero.
- **Secondary — per-K TOST.** A two-one-sided-tests equivalence test per K
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
    """Single-DOF trend test: mean per-image slope of interaction vs log2(K).

    Returns ``(mean_slope, lo, hi, p)``. The slope and its p-value are the
    primary result; the zero crossing is read from the per-K absolute
    interaction (this fit is linear in log2(K), so it is a poor model for the
    plateauing interaction and its own crossing point is not reported).
    """
    logKs = np.log2(Ks)
    slopes = []
    for image, by_K in per_image.items():
        ys = [by_K[K] for K in Ks if K in by_K]
        if len(ys) < 2:
            continue
        slope = np.polyfit(logKs[: len(ys)], ys, 1)[0]
        slopes.append(slope)
    slopes = np.asarray(slopes)
    mean_slope = float(slopes.mean())
    _, p = stats.ttest_1samp(slopes, 0.0)
    lo, hi = stats.t.interval(0.95, len(slopes) - 1,
                              loc=mean_slope, scale=stats.sem(slopes))
    return mean_slope, float(lo), float(hi), float(p)


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

        mean_slope, lo, hi, p = _trend_test(inter_matrix, Ks)
        print(f"  PRIMARY trend test: mean per-image slope = {mean_slope:+.4f}/doubling "
              f"95% CI [{lo:+.4f}, {hi:+.4f}]  p = {p:.3g} "
              f"(zero crossing read from the per-K absolute interaction, not this fit)")

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
