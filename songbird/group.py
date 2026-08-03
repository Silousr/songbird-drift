"""Group-level comparison: did the treated birds change more than the controls?

Everything else in this toolkit answers "did *this* bird change". An experiment asks
whether a manipulation had an effect, which is a different question with a different unit
of analysis.

**The bird is the unit of replication.** Not the syllable, not the day, not the bout. The
functions here accept exactly one number per bird, and that restriction is deliberate:
pooling syllables across birds and testing on the pool is pseudo-replication. It treats one
bird's thousands of correlated renditions as thousands of independent observations, and the
resulting p-value can be orders of magnitude too small. Compute a drift value per bird
first — with :func:`songbird.drift.bootstrap_drift_ci` or
:func:`songbird.drift.bootstrap_dispersion_ci` — then bring those here.

**Permutation, not a t-test.** Songbird experiments typically run 5–15 birds per group, and
the drift statistic is right-skewed. A test resting on the normality of the mean is not
safe at that sample size; shuffling group labels assumes nothing beyond exchangeability
under the null.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = ["GroupComparison", "birds_needed", "compare_groups"]

ALTERNATIVES = ("two-sided", "greater", "less")


@dataclass(frozen=True)
class GroupComparison:
    """Result of a treated-versus-control comparison."""

    observed_difference: float
    p_value: float
    ci_low: float
    ci_high: float
    n_treated: int
    n_control: int
    alternative: str
    between_bird_sd: float
    within_bird_sd: float | None = None
    limiting_factor: str | None = None

    def summary(self) -> str:
        lines = [
            f"treated n={self.n_treated}, control n={self.n_control}",
            f"difference {self.observed_difference:+.4f} "
            f"[{self.ci_low:+.4f}, {self.ci_high:+.4f}]",
            f"permutation p = {self.p_value:.4f} ({self.alternative})",
            f"between-bird SD {self.between_bird_sd:.4f}",
        ]
        if self.limiting_factor:
            lines.append(
                f"limited by {self.limiting_factor} variation "
                f"(within-bird SD {self.within_bird_sd:.4f})"
            )
        return "\n".join(lines)


def _permutation_p(
    treated: np.ndarray, control: np.ndarray, alternative: str,
    n_permutations: int, seed: int,
) -> float:
    observed = treated.mean() - control.mean()
    pooled = np.concatenate([treated, control])
    n_treated = len(treated)
    total = len(pooled)

    # With few birds the permutation distribution is small enough to enumerate exactly,
    # which removes Monte-Carlo noise from the p-value entirely.
    from math import comb

    if comb(total, n_treated) <= max(n_permutations, 1):
        differences = np.array([
            pooled[list(idx)].mean()
            - pooled[list(set(range(total)) - set(idx))].mean()
            for idx in combinations(range(total), n_treated)
        ])
    else:
        rng = np.random.default_rng(seed)
        differences = np.empty(n_permutations)
        for i in range(n_permutations):
            shuffled = rng.permutation(pooled)
            differences[i] = shuffled[:n_treated].mean() - shuffled[n_treated:].mean()

    if alternative == "greater":
        extreme = differences >= observed
    elif alternative == "less":
        extreme = differences <= observed
    else:
        extreme = np.abs(differences) >= abs(observed)

    # (count + 1) / (n + 1): a permutation p-value is never exactly zero, because the
    # observed labelling is itself one of the arrangements under the null.
    return float((extreme.sum() + 1) / (len(differences) + 1))


def compare_groups(
    treated: np.ndarray,
    control: np.ndarray,
    alternative: str = "two-sided",
    n_permutations: int = 10_000,
    within_bird_sd: float | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> GroupComparison:
    """Compare per-bird drift between a treated and a control group.

    ``treated`` and ``control`` hold **one value per bird**. Pass ``within_bird_sd`` — the
    typical width of a single bird's drift estimate — to get a diagnostic saying whether
    the experiment is limited by recording volume or by biological variability between
    birds. That distinction decides whether to record more per bird or to add birds, and
    they are not interchangeable.
    """
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")

    treated = np.asarray(treated, dtype=float)
    control = np.asarray(control, dtype=float)
    if len(treated) < 2 or len(control) < 2:
        raise ValueError(
            f"need at least 2 birds per group; got {len(treated)} treated and "
            f"{len(control)} control"
        )

    observed = float(treated.mean() - control.mean())
    p_value = _permutation_p(treated, control, alternative, n_permutations, seed)

    rng = np.random.default_rng(seed + 1)
    draws = [
        rng.choice(treated, len(treated), replace=True).mean()
        - rng.choice(control, len(control), replace=True).mean()
        for _ in range(n_boot)
    ]
    ci_low, ci_high = np.percentile(draws, [2.5, 97.5])

    between = float(np.std(np.concatenate([
        treated - treated.mean(), control - control.mean()
    ]), ddof=1))

    limiting = None
    if within_bird_sd is not None:
        # Compare the noise on one bird's estimate against the spread between birds.
        limiting = "within-bird" if within_bird_sd > between else "between-bird"

    return GroupComparison(
        observed_difference=observed, p_value=p_value,
        ci_low=float(ci_low), ci_high=float(ci_high),
        n_treated=len(treated), n_control=len(control), alternative=alternative,
        between_bird_sd=between, within_bird_sd=within_bird_sd,
        limiting_factor=limiting,
    )


def birds_needed(
    effect: float,
    between_bird_sd: float,
    alpha: float = 0.05,
    power: float = 0.8,
    alternative: str = "greater",
    max_n: int = 40,
    n_simulations: int = 400,
    n_permutations: int = 2000,
    seed: int = 0,
) -> float:
    """Birds **per group** needed to detect ``effect`` at the requested power.

    Estimated by simulating the same permutation test that would actually be run, rather
    than from a normal-theory formula — at 5–15 birds per group the two disagree, and the
    formula is the optimistic one.

    Returns ``nan`` if ``max_n`` birds per group still fall short, rather than returning
    ``max_n`` and implying it would suffice.
    """
    rng = np.random.default_rng(seed)
    for n in range(3, max_n + 1):
        detections = 0
        for _ in range(n_simulations):
            treated = rng.normal(effect, between_bird_sd, n)
            control = rng.normal(0.0, between_bird_sd, n)
            result = _permutation_p(treated, control, alternative,
                                    n_permutations, int(rng.integers(1 << 30)))
            detections += result < alpha
        if detections / n_simulations >= power:
            return float(n)
    return float("nan")
