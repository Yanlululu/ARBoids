"""Paired episode and training-seed inference; no episode pseudoreplication."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest


def exact_sign_flip(differences):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 1 <= len(values) <= 16:
        raise ValueError("Expected one finite paired difference per training seed (1..16)")
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=len(values))))
    null = np.abs(signs @ values / len(values))
    observed = abs(values.mean())
    return float(np.mean(null >= observed - 1e-14))


def holm(pvalues):
    values = np.asarray(pvalues, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("Invalid p-values")
    order = np.argsort(values)
    adjusted = np.minimum(1., np.maximum.accumulate(values[order] * np.arange(len(values), 0, -1)))
    output = np.empty_like(adjusted)
    output[order] = adjusted
    return output.tolist()


def paired_fixed(rows, reference, outcome):
    left = {r["seed"]: r for r in rows}
    right = {r["seed"]: r for r in reference}
    if len(left) != len(rows) or len(right) != len(reference) or left.keys() != right.keys():
        raise ValueError("Paired evaluations require the same unique scenario seeds")
    for seed in left:
        if left[seed].get("initial_sha256") != right[seed].get("initial_sha256"):
            raise ValueError("Paired policies did not receive the same initial condition")
    a = np.asarray([left[s]["outcome_code"] == outcome for s in sorted(left)], dtype=int)
    b = np.asarray([right[s]["outcome_code"] == outcome for s in sorted(left)], dtype=int)
    improved = int(np.sum((a == 0) & (b == 1)))
    worsened = int(np.sum((a == 1) & (b == 0)))
    discordant = improved + worsened
    p = binomtest(improved, discordant, .5).pvalue if discordant else 1.
    return dict(episodes=len(a), candidate_events=int(a.sum()), reference_events=int(b.sum()),
                difference=float((a - b).mean()), candidate_only=worsened, reference_only=improved,
                exact_mcnemar_two_sided_p=float(p), unit="scenario for these two fixed checkpoints")


def crossed_interval(differences, repetitions=4000, seed=271828):
    values = np.asarray(differences, dtype=float)
    if values.ndim != 2 or min(values.shape) < 2:
        raise ValueError("Expected training seeds x common evaluation scenarios")
    rng = np.random.default_rng(seed)
    estimates = np.empty(repetitions)
    for i in range(repetitions):
        r = rng.integers(values.shape[0], size=values.shape[0])
        c = rng.integers(values.shape[1], size=values.shape[1])
        estimates[i] = values[r[:, None], c].mean()
    return [float(x) for x in np.quantile(estimates, (.025, .975))]


def seed_comparison(candidate, reference):
    a, b = np.asarray(candidate, dtype=float), np.asarray(reference, dtype=float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("Matched training seeds and evaluation scenarios are required")
    delta = a - b
    return dict(training_seeds=len(a), scenarios_per_seed=a.shape[1],
                candidate_rate=float(a.mean()), reference_rate=float(b.mean()),
                paired_difference=float(delta.mean()), per_seed_differences=delta.mean(1).tolist(),
                crossed_bootstrap_95_interval=crossed_interval(delta),
                exact_paired_sign_flip_two_sided_p=exact_sign_flip(delta.mean(1)),
                unit="independent training seed; common scenarios resampled as a crossed factor")


def power_sensitivity(repetitions=10000):
    """Planning sensitivity, not observed power or an assumed true effect."""
    n, alpha = 8, .025  # conservative first Holm threshold for two primary contrasts
    rng = np.random.default_rng(1234567)
    signs = np.asarray(list(itertools.product((-1., 1.), repeat=n)))
    rows = []
    for dz in (.5, .8, 1., 1.25, 1.5):
        detected = 0
        for _ in range(repetitions // 100):
            differences = rng.normal(dz, 1., size=(100, n))
            null = np.abs(differences @ signs.T / n)
            p = (null >= np.abs(differences.mean(1))[:, None] - 1e-14).mean(1)
            detected += int((p <= alpha).sum())
        count = repetitions // 100 * 100
        estimate = detected / count
        interval = binomtest(detected, count).proportion_ci()
        rows.append(dict(standardized_paired_effect=dz, power=estimate,
                         monte_carlo_95_interval=[interval.low, interval.high]))
    return dict(design="8 independent paired training seeds", alpha_per_comparison=alpha,
                test="two-sided exact sign flip of seed-level differences",
                assumptions="Independent normal paired seed differences with unit SD; sensitivity, not an estimated effect",
                repetitions=repetitions, rng_seed=1234567, results=rows,
                interpretation="Eight seeds may be insufficient for small effects. More episodes cannot replace more training seeds.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--power-output", type=Path, required=True)
    args = parser.parse_args()
    output = power_sensitivity()
    args.power_output.parent.mkdir(parents=True, exist_ok=True)
    args.power_output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(output, indent=2))
