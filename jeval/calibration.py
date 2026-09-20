"""Calibration measurement: binning, metrics, and interval estimation.

Every function here is pure and deterministic given its inputs. The synthetic-data tests in
``tests/test_synth.py`` exist to prove these functions restore a known miscalibration, so any
change to this module has to keep those tests green.
"""

from __future__ import annotations

from collections.abc import Sequence
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
