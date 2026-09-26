"""Calibration measurement: binning, metrics, and interval estimation.

Every function here is pure and deterministic given its inputs. The synthetic-data tests in
``tests/test_synth.py`` exist to prove these functions restore a known miscalibration, so any
change to this module has to keep those tests green.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray

DEFAULT_N_BINS = 10
MIN_BIN_SIZE = 20
DEFAULT_BOOTSTRAP_SAMPLES = 1000
DEFAULT_ALPHA = 0.05


@dataclass(frozen=True)
class CalibrationBin:
    """One confidence bin with its observed accuracy and Wilson interval."""

    index: int
    lo: float
    hi: float
    n: int
    mean_confidence: float
    accuracy: float
    ci_low: float
    ci_high: float

    @property
    def gap(self) -> float:
        """Accuracy minus mean confidence. Negative means overconfidence."""
        return self.accuracy - self.mean_confidence

    @property
    def label(self) -> str:
        if abs(self.hi - self.lo) < 1e-12:
            return f"{self.lo:.2f}"
        return f"{self.lo:.2f}-{self.hi:.2f}"


@dataclass(frozen=True)
class CalibrationMetrics:
    """Calibration metrics for one population of labeled decisions."""

    n: int
    ece: float
    mce: float
    brier: float
    ece_ci_low: float
    ece_ci_high: float
    bins: tuple[CalibrationBin, ...]
    bins_requested: int
    binning: str
    confidence_buckets: tuple[tuple[float, float, int], ...] = ()

    @property
    def ece_ci_span(self) -> float:
        return self.ece_ci_high - self.ece_ci_low

    @property
    def worst_bin(self) -> CalibrationBin | None:
        """The bin that best supports a claim, preferring bins with enough samples.

        A four-record bin with a huge gap is not evidence of miscalibration, it is evidence of
        four records. Bins below a small-sample floor are only used when nothing else exists,
        so the one-line diagnosis does not get hijacked by noise.
        """
        scored = [b for b in self.bins if b.n > 0]
        if not scored:
            return None
        floor = max(5, int(0.02 * self.n))
        candidates = [b for b in scored if b.n >= floor] or scored
        return max(candidates, key=lambda b: abs(b.gap))


def normal_quantile(alpha: float) -> float:
    """Two-sided normal quantile for a confidence level of ``1 - alpha``."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    return NormalDist().inv_cdf(1.0 - alpha / 2.0)


def wilson_interval(successes: int, n: int, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because bin sizes in real decision logs are
    routinely small enough for the approximation to break down.
    """
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError("successes must satisfy 0 <= successes <= n")
    z = normal_quantile(alpha)
    p = successes / n
    denom = 1.0 + z * z / n
    center = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (float(max(0.0, center - half)), float(min(1.0, center + half)))


def choose_n_bins(n: int, requested: int = DEFAULT_N_BINS, min_bin_size: int = MIN_BIN_SIZE) -> int:
    """Shrink the requested bin count so no bin is starved of samples."""
    if n <= 0:
        return 0
    if requested < 1:
        raise ValueError("requested bins must be >= 1")
    affordable = max(1, n // max(1, min_bin_size))
    return max(1, min(requested, affordable))


def quantile_edges(values: NDArray[np.float64], n_bins: int) -> list[float]:
    """Equal-frequency bin edges.

    Quantile binning is the default because real confidence distributions pile up near the
    top: equal-width bins leave most bins empty and one bin holding almost everything.
    """
    if values.size == 0:
        return []
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    raw = np.quantile(values, quantiles)
    edges: list[float] = [float(raw[0])]
    for edge in raw[1:]:
        value = float(edge)
        if value > edges[-1]:
            edges.append(value)
    if len(edges) == 1:
        # A single distinct confidence value: one degenerate bin that still covers it.
        edges.append(edges[0])
    return edges


def equal_width_edges(values: NDArray[np.float64], n_bins: int) -> list[float]:
    """Equal-width bin edges spanning the observed confidence range."""
    if values.size == 0:
        return []
    lo = float(values.min())
    hi = float(values.max())
    if abs(hi - lo) < 1e-12:
        return [lo, hi]
    return [float(edge) for edge in np.linspace(lo, hi, n_bins + 1)]


def bins_from_edges(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    edges: Sequence[float],
    alpha: float = DEFAULT_ALPHA,
) -> tuple[CalibrationBin, ...]:
    """Bucket observations into bins and attach Wilson intervals."""
    if len(edges) < 2:
        return ()
    assigned = np.clip(np.digitize(confidences, list(edges[1:-1]), right=False), 0, len(edges) - 2)
    bins: list[CalibrationBin] = []
    for index in range(len(edges) - 1):
        mask = assigned == index
        n = int(mask.sum())
        if n == 0:
            continue
        successes = int(correct[mask].sum())
        ci_low, ci_high = wilson_interval(successes, n, alpha)
        bins.append(
            CalibrationBin(
                index=index,
                lo=float(edges[index]),
                hi=float(edges[index + 1]),
                n=n,
                mean_confidence=float(confidences[mask].mean()),
                accuracy=successes / n,
                ci_low=ci_low,
                ci_high=ci_high,
            )
        )
    return tuple(bins)


def expected_calibration_error(bins: Sequence[CalibrationBin], n: int) -> float:
    """Weighted mean absolute gap between confidence and accuracy."""
    if n <= 0:
        return float("nan")
    return float(sum(b.n / n * abs(b.gap) for b in bins))


def maximum_calibration_error(bins: Sequence[CalibrationBin]) -> float:
    """Largest absolute gap over bins that contain samples."""
    if not bins:
        return float("nan")
    return float(max(abs(b.gap) for b in bins))


def confidence_buckets(
    confidences: NDArray[np.float64], n_buckets: int = 20
) -> tuple[tuple[float, float, int], ...]:
    """Counts of confidences over equal-width buckets on [0, 1].

    Equal width is right here and wrong for the calibration bins: a density view should show
    where the volume actually sits, and quantile buckets would flatten it by construction.
    """
    if confidences.size == 0:
        return ()
    counts, edges = np.histogram(np.clip(confidences, 0.0, 1.0), bins=n_buckets, range=(0.0, 1.0))
    return tuple((float(edges[i]), float(edges[i + 1]), int(counts[i])) for i in range(len(counts)))


def brier_score(confidences: NDArray[np.float64], correct: NDArray[np.bool_]) -> float:
    """Mean squared error of the confidence against the outcome."""
    if confidences.size == 0:
        return float("nan")
    return float(np.mean((confidences - correct.astype(float)) ** 2))


def bootstrap_ece_ci(
    confidences: NDArray[np.float64],
    correct: NDArray[np.bool_],
    edges: Sequence[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> tuple[float, float]:
    """Percentile bootstrap interval for ECE with the bins held fixed.

    Bins stay fixed across resamples so the interval measures sampling noise in the accuracy,
    not movement of the bin edges themselves.
    """
    if confidences.size == 0:
        return (float("nan"), float("nan"))
    if n_boot <= 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    n = confidences.size
    estimates = np.empty(n_boot, dtype=float)
    for draw in range(n_boot):
        idx = rng.integers(0, n, size=n)
        bins = bins_from_edges(confidences[idx], correct[idx], edges, alpha)
        estimates[draw] = expected_calibration_error(bins, n)
    observed = expected_calibration_error(bins_from_edges(confidences, correct, edges, alpha), n)
    spread = float(np.ptp(estimates))
    if spread <= 0.0:
        # Every resample agreed exactly: an all-identical draw says nothing about sampling error,
        # and reporting [x, x] would assert a precision the sample does not have.
        return (float("nan"), float("nan"))
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    # |accuracy - confidence| is convex, so the resample ECE is biased upward and the percentile
    # interval can sit entirely above the value it is an interval for — and above zero for a
    # calibrated log. A basic-bootstrap recentering is not the answer either: ECE cannot be
    # negative, and recentering drives the bound below zero on exactly those samples. So the
    # interval is widened to contain its own point estimate, which is the least a reported
    # interval owes its reader.
    return (min(lower, observed), max(upper, observed))


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value on the discordant pairs of a paired comparison.

    ``b`` counts pairs the baseline got right and the current version got wrong, ``c`` the
    reverse. Concordant pairs carry no information about which version is more accurate, so under
    the null hypothesis each discordant pair is a fair coin: the p-value is twice the smaller
    binomial tail of ``Binomial(b + c, 0.5)``, capped at 1. With no discordant pair at all there
    is nothing to test and the p-value is 1.
    """
    if b < 0 or c < 0:
        raise ValueError("discordant counts must be >= 0")
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(b, c) + 1))
    return float(min(1.0, 2 * tail / 2**n))


@dataclass(frozen=True)
class PairedDifference:
    """One metric on both sides of a paired comparison, and ``current - baseline``."""

    baseline: float
    current: float
    difference: float
    ci_low: float
    ci_high: float


@dataclass(frozen=True)
class PairedComparison:
    """A head-to-head comparison of two versions on the same ``n`` requests."""

    n: int
    accuracy: PairedDifference
    ece: PairedDifference
    brier: PairedDifference
    discordant_baseline_only: int
    discordant_current_only: int
    mcnemar_p: float


def _percentile_interval(
    estimates: NDArray[np.float64], alpha: float, observed: float
) -> tuple[float, float]:
    if float(np.ptp(estimates)) <= 0.0:
        # Every resample agreed exactly, as it does for two identical logs: that says nothing about
        # sampling error, and [x, x] would assert a precision the sample does not have.
        return (float("nan"), float("nan"))
    lower = float(np.quantile(estimates, alpha / 2.0))
    upper = float(np.quantile(estimates, 1.0 - alpha / 2.0))
    return (min(lower, observed), max(upper, observed))


def paired_bootstrap(
    baseline_confidences: NDArray[np.float64],
    baseline_correct: NDArray[np.bool_],
    current_confidences: NDArray[np.float64],
    current_correct: NDArray[np.bool_],
    edges: Sequence[float],
    *,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    alpha: float = DEFAULT_ALPHA,
    seed: int = 0,
) -> PairedComparison:
    """Bootstrap ``current - baseline`` for accuracy, ECE and Brier over pairs, plus McNemar.

    Index ``i`` of the four arrays is one request answered by both versions. A resample draws pair
    indices, so both sides of a pair always travel together and the difference is measured on the
    same requests — that is what removes request difficulty from the comparison. ECE on each side is
    computed over the same fixed ``edges`` in every resample, for the reason ``bootstrap_ece_ci``
    holds its bins fixed. Intervals are percentile intervals at ``alpha``, widened to contain their
    own point estimate as ``bootstrap_ece_ci`` does; an interval with no spread at all is NaN.
    """
    n = int(baseline_confidences.size)
    if not (baseline_correct.size == current_confidences.size == current_correct.size == n):
        raise ValueError("paired arrays must all have the same length")
    if n == 0:
        raise ValueError("a paired comparison needs at least one pair")

    def measure(idx: NDArray[np.intp]) -> tuple[float, float, float, float, float, float]:
        bc, bh = baseline_confidences[idx], baseline_correct[idx]
        cc, ch = current_confidences[idx], current_correct[idx]
        return (
            float(bh.mean()),
            float(ch.mean()),
            expected_calibration_error(bins_from_edges(bc, bh, edges, alpha), n),
            expected_calibration_error(bins_from_edges(cc, ch, edges, alpha), n),
            brier_score(bc, bh),
            brier_score(cc, ch),
        )

    observed = measure(np.arange(n))
    draws = np.empty((max(0, n_boot), 3), dtype=float)
    rng = np.random.default_rng(seed)
    for draw in range(max(0, n_boot)):
        values = measure(rng.integers(0, n, size=n))
        draws[draw] = (values[1] - values[0], values[3] - values[2], values[5] - values[4])

    differences: list[PairedDifference] = []
    for column in range(3):
        before, after = observed[2 * column], observed[2 * column + 1]
        difference = after - before
        interval = (
            _percentile_interval(draws[:, column], alpha, difference)
            if n_boot > 0
            else (float("nan"), float("nan"))
        )
        differences.append(PairedDifference(before, after, difference, interval[0], interval[1]))

    b = int(np.sum(baseline_correct & ~current_correct))
    c = int(np.sum(~baseline_correct & current_correct))
    return PairedComparison(
        n=n,
        accuracy=differences[0],
        ece=differences[1],
        brier=differences[2],
        discordant_baseline_only=b,
        discordant_current_only=c,
        mcnemar_p=mcnemar_exact(b, c),
    )


def compute_calibration(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
    *,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> CalibrationMetrics:
    """Measure calibration of ``(confidence, correct)`` pairs."""
    conf = np.asarray(list(confidences), dtype=float)
    hit = np.asarray(list(correct), dtype=bool)
    if conf.size != hit.size:
        raise ValueError("confidences and correct must have the same length")
    n = int(conf.size)
    binning = "equal-width" if equal_width else "quantile"
    if n == 0:
        return CalibrationMetrics(
            n=0,
            ece=float("nan"),
            mce=float("nan"),
            brier=float("nan"),
            ece_ci_low=float("nan"),
            ece_ci_high=float("nan"),
            bins=(),
            bins_requested=n_bins,
            binning=binning,
        )
    effective_bins = choose_n_bins(n, n_bins)
    edges = (
        equal_width_edges(conf, effective_bins)
        if equal_width
        else quantile_edges(conf, effective_bins)
    )
    bins = bins_from_edges(conf, hit, edges, alpha)
    buckets = confidence_buckets(conf)
    ece = expected_calibration_error(bins, n)
    ci_low, ci_high = bootstrap_ece_ci(conf, hit, edges, n_boot=n_boot, alpha=alpha, seed=seed)
    return CalibrationMetrics(
        n=n,
        ece=ece,
        mce=maximum_calibration_error(bins),
        brier=brier_score(conf, hit),
        ece_ci_low=ci_low,
        ece_ci_high=ci_high,
        bins=bins,
        bins_requested=n_bins,
        binning=binning,
        confidence_buckets=buckets,
    )


def segment_distribution(values: Sequence[float], edges: Sequence[float]) -> NDArray[np.float64]:
    """Histogram of ``values`` over ``edges`` (used for score-type level distributions)."""
    if len(edges) < 2 or not len(values):
        return np.zeros(max(0, len(edges) - 1), dtype=float)
    counts, _ = np.histogram(np.asarray(list(values), dtype=float), bins=list(edges))
    shares = np.asarray(counts, dtype=float)
    total = float(shares.sum())
    if total == 0:
        return shares
    return np.asarray(shares / total, dtype=float)


def diagnose(metrics: CalibrationMetrics, tolerance: float = 0.02) -> str:
    """One-line, plain-language reading of a calibration result."""
    if metrics.n == 0:
        return "No labeled decisions: nothing to measure yet."
    worst = metrics.worst_bin
    if worst is None or metrics.mce <= tolerance:
        return (
            f"Well calibrated within sampling error: ECE {metrics.ece:.3f} "
            f"(95% CI {metrics.ece_ci_low:.3f}-{metrics.ece_ci_high:.3f}, n={metrics.n})."
        )
    interval = f"{worst.ci_low:.2f}-{worst.ci_high:.2f}"
    if worst.ci_low <= worst.mean_confidence <= worst.ci_high:
        # The bin's own interval contains the confidence it is supposed to contradict, so a
        # direction cannot be claimed from it however large the gap looks.
        return (
            f"Not distinguishable from calibrated at this n: the worst bin ({worst.label}) claims "
            f"{worst.mean_confidence:.2f} and observed {worst.accuracy:.2f} "
            f"(95% CI {interval}, n={worst.n}); ECE {metrics.ece:.3f}."
        )
    direction = "overconfidence" if worst.gap < 0 else "underconfidence"
    return (
        f"{direction.capitalize()} in {worst.label}: claimed {worst.mean_confidence:.2f}, "
        f"observed {worst.accuracy:.2f} (95% CI {interval}, n={worst.n}); "
        f"ECE {metrics.ece:.3f}."
    )


# --------------------------------------------------------------------------------------
# Discrimination: does confidence rank the model's own errors?
# --------------------------------------------------------------------------------------
# Calibration and discrimination are different questions. A model that says 0.8 on every answer
# and is right 80% of the time is perfectly calibrated and ranks nothing: no threshold can buy
# accuracy from it, because escalating its "least confident" answers escalates a random sample.


@dataclass(frozen=True)
class DiscriminationMetrics:
    """How well confidence separates right answers from wrong ones, with the risk-coverage curve.

    ``levels``, ``coverage`` and ``risk`` are parallel, most confident first: automating every
    answer stated at or above ``levels[i]`` covers ``coverage[i]`` of the decisions, and
    ``risk[i]`` of those are wrong. Tied confidences form one step, since a threshold cannot run
    one 0.8 and escalate another.
    """

    n: int
    n_wrong: int
    auroc: float
    auroc_ci_low: float
    auroc_ci_high: float
    aurc: float
    aurc_ci_low: float
    aurc_ci_high: float
    full_coverage_risk: float
    levels: tuple[float, ...] = ()
    coverage: tuple[float, ...] = ()
    risk: tuple[float, ...] = ()

    @property
    def constant(self) -> bool:
        """Every decision carries the same confidence, so there is no ranking to measure."""
        return self.n > 0 and len(self.levels) == 1

    @property
    def perfect_aurc(self) -> float:
        """The area a perfect ranking would leave: every right answer before every wrong one."""
        if self.n == 0:
            return float("nan")
        n_right = self.n - self.n_wrong
        return float(sum(max(0, k - n_right) / k for k in range(1, self.n + 1)) / self.n)

    def at_threshold(self, threshold: float) -> tuple[float, float]:
        """``(coverage, risk)`` when every answer stated at or above ``threshold`` is automated."""
        chosen = -1
        for index, level in enumerate(self.levels):
            if level >= threshold:
                chosen = index
        if chosen < 0:
            return (0.0, float("nan"))
        return (self.coverage[chosen], self.risk[chosen])

    def at_coverage(self, share: float) -> tuple[float, float, float]:
        """``(level, coverage, risk)`` at the first step that automates at least ``share``."""
        for level, covered, risk in zip(self.levels, self.coverage, self.risk, strict=True):
            if covered >= share - 1e-12:
                return (level, covered, risk)
        return (float("nan"), float("nan"), float("nan"))


def _as_pairs(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    conf = np.asarray(list(confidences), dtype=float)
    hit = np.asarray(list(correct), dtype=bool)
    if conf.size != hit.size:
        raise ValueError("confidences and correct must have the same length")
    return conf, hit


def _auroc(conf: NDArray[np.float64], hit: NDArray[np.bool_]) -> float:
    """Mann-Whitney AUROC with tied confidences given their average rank (a tie counts 1/2)."""
    n_right = int(hit.sum())
    n_wrong = int(hit.size - n_right)
    if n_right == 0 or n_wrong == 0:
        return float("nan")
    _, inverse, counts = np.unique(conf, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts).astype(float)
    average_rank = ends - (counts.astype(float) - 1.0) / 2.0
    ranks = average_rank[inverse]
    u_stat = float(ranks[hit].sum()) - n_right * (n_right + 1) / 2.0
    return u_stat / (n_right * n_wrong)


def _risk_coverage(
    conf: NDArray[np.float64], hit: NDArray[np.bool_]
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """``(levels, coverage, risk, step_share)``, most confident first, one step per value."""
    if conf.size == 0:
        empty = np.zeros(0, dtype=float)
        return empty, empty, empty, empty
    levels, inverse, counts = np.unique(conf, return_inverse=True, return_counts=True)
    errors = np.bincount(inverse, weights=(~hit).astype(float), minlength=levels.size)
    levels, counts, errors = levels[::-1], counts[::-1].astype(float), errors[::-1]
    covered = np.cumsum(counts)
    risk = np.cumsum(errors) / covered
    n = float(conf.size)
    return (
        np.asarray(levels, dtype=float),
        np.asarray(covered / n, dtype=float),
        np.asarray(risk, dtype=float),
        np.asarray(counts / n, dtype=float),
    )


def _aurc(conf: NDArray[np.float64], hit: NDArray[np.bool_]) -> float:
    _, _, risk, share = _risk_coverage(conf, hit)
    if risk.size == 0:
        return float("nan")
    return float(np.sum(share * risk))


def auroc(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
) -> float:
    """Probability that a right answer carries more confidence than a wrong one.

    0.5 is a coin flip -- confidence ranks nothing -- and 1.0 is a perfect ranking. Undefined
    (``nan``) when every answer is right or every answer is wrong: there is nothing to separate.
    """
    conf, hit = _as_pairs(confidences, correct)
    return _auroc(conf, hit)


def risk_coverage(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """``(coverage, risk)``: the error rate among the automated answers at every threshold.

    Coverage is the share of decisions automated when every answer at or above a confidence runs;
    risk is the share of those that are wrong. Tied confidences move together.
    """
    conf, hit = _as_pairs(confidences, correct)
    _, coverage, risk, _ = _risk_coverage(conf, hit)
    return coverage, risk


def aurc(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
) -> float:
    """Area under the risk-coverage curve: the mean error rate over every coverage level.

    Without ties this is ``(1/n) * sum(risk_k)``; a tied step counts once per decision in it. If
    confidence ranked nothing, the expected area would equal the error rate at full coverage.
    """
    conf, hit = _as_pairs(confidences, correct)
    return _aurc(conf, hit)


def _percentile_ci(
    estimates: NDArray[np.float64], observed: float, alpha: float
) -> tuple[float, float]:
    """Percentile interval widened to contain its own point estimate (see ``bootstrap_ece_ci``)."""
    finite = estimates[np.isfinite(estimates)]
    # A resample that drew only right (or only wrong) answers has no AUROC; if that is most of
    # them, the sample is too lopsided for an interval to mean anything.
    if finite.size < max(2, estimates.size // 2) or float(np.ptp(finite)) <= 0.0:
        return (float("nan"), float("nan"))
    lower = float(np.quantile(finite, alpha / 2.0))
    upper = float(np.quantile(finite, 1.0 - alpha / 2.0))
    if observed != observed:
        return (lower, upper)
    return (min(lower, observed), max(upper, observed))


#: The fewest wrong (or right) answers an AUROC interval is computed from; the report's bin floor.
MIN_DISCRIMINATION_CLASS = MIN_BIN_SIZE


def compute_discrimination(
    confidences: Sequence[float] | NDArray[np.float64],
    correct: Sequence[bool] | NDArray[np.bool_],
    *,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
) -> DiscriminationMetrics:
    """AUROC and AURC with percentile bootstrap intervals, plus the risk-coverage curve."""
    conf, hit = _as_pairs(confidences, correct)
    n = int(conf.size)
    nan = float("nan")
    if n == 0:
        return DiscriminationMetrics(
            n=0,
            n_wrong=0,
            auroc=nan,
            auroc_ci_low=nan,
            auroc_ci_high=nan,
            aurc=nan,
            aurc_ci_low=nan,
            aurc_ci_high=nan,
            full_coverage_risk=nan,
        )
    levels, coverage, risk, _ = _risk_coverage(conf, hit)
    observed_auroc = _auroc(conf, hit)
    observed_aurc = _aurc(conf, hit)
    auroc_ci: tuple[float, float] = (nan, nan)
    aurc_ci: tuple[float, float] = (nan, nan)
    n_wrong = int((~hit).sum())
    # With a handful of wrong (or right) answers the resampled AUROC is lumpy, and a percentile
    # interval on it excluded 0.5 up to 42% of the time on data where confidence ranked nothing.
    # Below the report's smallest bin on either side there is no interval, and the reading says so.
    if n_boot > 0 and min(n_wrong, n - n_wrong) >= MIN_DISCRIMINATION_CLASS:
        rng = np.random.default_rng(seed)
        auroc_draws = np.empty(n_boot, dtype=float)
        aurc_draws = np.empty(n_boot, dtype=float)
        for draw in range(n_boot):
            idx = rng.integers(0, n, size=n)
            auroc_draws[draw] = _auroc(conf[idx], hit[idx])
            aurc_draws[draw] = _aurc(conf[idx], hit[idx])
        auroc_ci = _percentile_ci(auroc_draws, observed_auroc, alpha)
        aurc_ci = _percentile_ci(aurc_draws, observed_aurc, alpha)
    return DiscriminationMetrics(
        n=n,
        n_wrong=n_wrong,
        auroc=observed_auroc,
        auroc_ci_low=auroc_ci[0],
        auroc_ci_high=auroc_ci[1],
        aurc=observed_aurc,
        aurc_ci_low=aurc_ci[0],
        aurc_ci_high=aurc_ci[1],
        full_coverage_risk=float(risk[-1]),
        levels=tuple(float(value) for value in levels),
        coverage=tuple(float(value) for value in coverage),
        risk=tuple(float(value) for value in risk),
    )


def describe_discrimination(metrics: DiscriminationMetrics) -> str:
    """One plain-language sentence on what the ranking can and cannot buy."""
    if metrics.n == 0:
        return "No labeled decisions: nothing to rank yet."
    if metrics.auroc != metrics.auroc:
        state = "right" if metrics.n_wrong == 0 else "wrong"
        return (
            f"Every one of the {metrics.n} labeled answers was {state}, so AUROC is undefined: "
            "there is nothing for confidence to separate. Label more decisions before reading "
            "anything into the ranking."
        )
    value = f"AUROC {metrics.auroc:.2f}"
    if metrics.constant:
        return (
            f"{value}: every decision carries the same confidence, so it ranks nothing; no "
            "threshold can buy accuracy here."
        )
    minority = min(metrics.n_wrong, metrics.n - metrics.n_wrong)
    if minority < MIN_DISCRIMINATION_CLASS:
        side = "wrong" if metrics.n_wrong <= metrics.n - metrics.n_wrong else "right"
        return (
            f"{value} (n={metrics.n}, only {minority} {side} answers): too few to tell whether "
            f"confidence ranks right above wrong; {MIN_DISCRIMINATION_CLASS} are needed before "
            "this number means anything."
        )
    has_interval = metrics.auroc_ci_low == metrics.auroc_ci_low
    interval = (
        f" (95% CI {metrics.auroc_ci_low:.2f}-{metrics.auroc_ci_high:.2f}, n={metrics.n})"
        if has_interval
        else f" (n={metrics.n}, no interval: too few right or wrong answers to resample)"
    )
    if has_interval and metrics.auroc_ci_low <= 0.5 <= metrics.auroc_ci_high:
        return (
            f"{value}{interval}: not distinguishable from a coin flip at this n. Confidence does "
            "not separate right from wrong answers here, so no threshold can buy accuracy."
        )
    if metrics.auroc < 0.5:
        return (
            f"{value}{interval}: inverted. The model is more confident on its wrong answers than "
            "its right ones, so raising the threshold escalates the wrong cases last."
        )
    share = f"a right answer outranks a wrong one {metrics.auroc:.0%} of the time"
    if metrics.auroc < 0.6:
        reading = "confidence barely separates right from wrong answers; a threshold buys little"
    elif metrics.auroc < 0.7:
        reading = "weak separation; a higher threshold buys some accuracy at a real cost in volume"
    elif metrics.auroc < 0.8:
        reading = "moderate separation; a higher threshold buys accuracy at a cost in volume"
    else:
        reading = "strong separation; escalating the least confident answers removes most errors"
    return f"{value}{interval}: {share}. {reading[0].upper()}{reading[1:]}."


# --------------------------------------------------------------------------------------
# Classwise calibration for ``choice`` questions
# --------------------------------------------------------------------------------------
# Top-1 calibration averages over whichever class was predicted, so a class that is overconfident
# only when it is the answer can hide behind classes that err the other way. Classwise ECE measures
# every class on its own probability, P(class = k) against (label == k), which needs the whole map.

MIN_CLASS_SIZE = 30  # the report's floor for a segment, a drift slice and a threshold sweep
MAP_SUM_TOLERANCE = 0.02


@dataclass(frozen=True)
class ClassCalibration:
    """One class, measured on ``(P(class), label == class)`` over every usable record.

    ``n`` is the number of usable records labeled with this class. ``metrics`` is ``None`` when
    the class was refused, and ``refused_reason`` says why.
    """

    name: str
    n: int
    metrics: CalibrationMetrics | None
    refused_reason: str = ""


@dataclass(frozen=True)
class ClasswiseMetrics:
    """Per-class calibration, the macro average, and what was set aside."""

    classes: tuple[ClassCalibration, ...]
    n_used: int
    n_refused_map: int
    macro_ece: float
    min_class: int = MIN_CLASS_SIZE

    @property
    def measured(self) -> tuple[ClassCalibration, ...]:
        return tuple(item for item in self.classes if item.metrics is not None)

    @property
    def worst(self) -> ClassCalibration | None:
        """The measured class with the largest ECE."""
        best: ClassCalibration | None = None
        for item in self.measured:
            if item.metrics is None:
                continue
            if best is None or best.metrics is None or item.metrics.ece > best.metrics.ece:
                best = item
        return best


def classwise_calibration(
    probabilities: Sequence[Mapping[str, float] | None],
    labels: Sequence[str],
    *,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 0,
    min_class: int = MIN_CLASS_SIZE,
    sum_tolerance: float = MAP_SUM_TOLERANCE,
) -> ClasswiseMetrics:
    """Classwise ECE over the records whose probability map is complete.

    A map that is missing, or does not sum to 1 within ``sum_tolerance``, is refused and counted:
    a top-k map says nothing about the classes it dropped, and reading them as zero would invent
    evidence. A class named in an accepted map but absent from another one gets probability 0
    there, which is what a map that sums to 1 asserts. A class with fewer than ``min_class``
    labeled records is refused, because its positives are too few to bin.
    """
    if len(probabilities) != len(labels):
        raise ValueError("probabilities and labels must have the same length")
    accepted: list[tuple[Mapping[str, float], str]] = []
    refused = 0
    for mapping, label in zip(probabilities, labels, strict=True):
        if not mapping or abs(float(sum(mapping.values())) - 1.0) > sum_tolerance:
            refused += 1
            continue
        accepted.append((mapping, label))
    names = sorted({name for mapping, _ in accepted for name in mapping})
    items: list[ClassCalibration] = []
    for name in names:
        stated = [float(mapping.get(name, 0.0)) for mapping, _ in accepted]
        positive = [label == name for _, label in accepted]
        support = sum(positive)
        if support < min_class:
            items.append(
                ClassCalibration(
                    name=name,
                    n=support,
                    metrics=None,
                    refused_reason=(
                        f"{support} labeled; fewer than {min_class} is too few to measure a class"
                    ),
                )
            )
            continue
        items.append(
            ClassCalibration(
                name=name,
                n=support,
                metrics=compute_calibration(
                    stated,
                    positive,
                    n_bins=n_bins,
                    equal_width=equal_width,
                    alpha=alpha,
                    n_boot=n_boot,
                    seed=seed,
                ),
            )
        )
    eces = [item.metrics.ece for item in items if item.metrics is not None]
    return ClasswiseMetrics(
        classes=tuple(items),
        n_used=len(accepted),
        n_refused_map=refused,
        macro_ece=float(np.mean(eces)) if eces else float("nan"),
        min_class=min_class,
    )


def describe_classwise(metrics: ClasswiseMetrics) -> str:
    """Name the worst class in one sentence, never claiming a ranking the intervals do not hold."""
    worst = metrics.worst
    if worst is None or worst.metrics is None:
        if metrics.n_used == 0:
            return (
                "No record carries a complete probability map, so no class can be measured on "
                "its own."
            )
        return (
            f"No class has {metrics.min_class} labeled decisions yet, so none can be measured on "
            "its own."
        )
    top = worst.metrics
    head = (
        f"Worst class: {worst.name}, ECE {top.ece:.3f} "
        f"(95% CI {top.ece_ci_low:.3f}-{top.ece_ci_high:.3f}, {worst.n} labeled)"
    )
    rivals = [
        item.metrics for item in metrics.measured if item is not worst and item.metrics is not None
    ]
    if not rivals:
        return head + "; it is the only class with enough labels to measure."
    runner_up = max(rival.ece_ci_high for rival in rivals)
    if top.ece_ci_low == top.ece_ci_low and top.ece_ci_low > runner_up:
        return head + "; its interval clears every other class's."
    return (
        head + "; its interval overlaps another class's, so the ranking is not settled at this n."
    )
