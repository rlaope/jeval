"""Drift detection: model-version and period comparison with CI exit codes.

The question here is not "did the model change" — that is a deploy event. It is "did a measured
decision boundary move when it did", and whether the movement is large enough to fail a build.

A **slice** is the unit of that measurement: a group of gold-labeled decision records that share
one ``model`` value, or, when grouping by period, one calendar period. Four rules are deliberate
and everything below follows from them:

1. **One edge set per comparison.** ``calibration.compute_calibration`` picks its bin edges from
   the sample it is handed, so calling it once per slice would move the bins between slices and
   part of any ECE difference would be a difference in binning rather than in the data. The edges
   are therefore fixed once, on the pooled population of the comparison, and every slice is
   measured over exactly that edge set.
2. **Small slices are omitted, never reported as a finding.** A slice below ``min_slice``
   gold-labeled records is dropped from the comparison and named in the view note. A movement
   measured on twelve records is not evidence.
3. **Gold labels only, and only what can be measured.** Silver labels are an agreement rate, not
   accuracy, and ``score`` records live on a different scale, so neither enters a slice: ``n``
   counts exactly the records an ECE can be computed from. A slice whose ``n`` is zero is kept
   with a NaN ECE rather than dropped, because "this model has no measurable evidence" is itself
   information.
4. **Checks never guess.** A check with no number to compare — an automation rate that was never
   computed because no cost matrix was applied — is skipped, not treated as a pass.

The workflow is ``split_models``/``split_by_period`` for the population views, ``compare`` for the
comparison, ``attach_thresholds`` to put a cost-derived threshold on each slice, ``parse_fail_on``
for the CI threshold spec, ``run_checks`` for the verdict, and ``format_ci_block`` for the text a
CI job prints. ``compare_paired`` is the head-to-head variant for shadow traffic, where both
versions answered the same requests: its docstring states the pairing rules.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Literal

import numpy as np

from jeval import calibration
from jeval.calibration import DEFAULT_N_BINS
from jeval.report.model import DriftFailure, DriftSlice, DriftView, ModelChange
from jeval.schema import DecisionRecord

if TYPE_CHECKING:  # pragma: no cover - imported for the type checker only
    from jeval.costs import CostAction

#: A slice needs at least this many measurable gold records to take part in a comparison.
DEFAULT_MIN_SLICE = 30
#: A question needs at least this many gold-labeled pairs before a paired comparison reports a
#: number. It matches ``DEFAULT_MIN_SLICE`` on purpose: a pair is one record on each side, so a
#: question that clears this bar also clears the unpaired comparison's bar on both sides.
MIN_PAIRS = 30
DEFAULT_ALPHA = 0.05
DEFAULT_N_BOOT = 200
DEFAULT_SEED = 0

#: Sweep grid resolution for an attached threshold; mirrors ``jeval.costs.DEFAULT_STEPS`` and is
#: kept as a literal so this module does not depend on the costs engine at import time.
DEFAULT_SWEEP_STEPS = 101

#: Written by :func:`attach_thresholds` into a view note whenever a cost matrix was applied, so
#: :func:`_threshold_lines` can tell "no cost matrix at all" apart from "a cost matrix whose sweep
#: withheld every threshold" without a new field on the view.
COST_MATRIX_NOTE = "cost matrix applied"

CHECK_ECE_INCREASE = "ece-increase"
CHECK_AUTO_RATE_DROP = "auto-rate-drop"
CHECK_ECE_ABOVE = "ece-above"

#: The only keys ``--fail-on`` accepts.
CHECK_KEYS: tuple[str, ...] = (CHECK_ECE_INCREASE, CHECK_AUTO_RATE_DROP, CHECK_ECE_ABOVE)

#: Accepted period groupings: ISO week, calendar month.
PERIOD_CODES: tuple[str, ...] = ("W", "M")

PeriodCode = Literal["W", "M"]


@dataclass(frozen=True)
class DriftCheck:
    """One ``--fail-on`` condition: a check key and the limit above which it fails."""

    key: str
    limit: float


@dataclass(frozen=True)
class _Unit:
    """One baseline/current pair, which is what a check and a table row operate on."""

    label: str
    before: DriftSlice
    after: DriftSlice


@dataclass(frozen=True)
class PairedQuestion:
    """One question of a paired comparison: its pairs, and either a comparison or a refusal."""

    question: str
    n_pairs: int
    comparison: calibration.PairedComparison | None = None
    refusal: str = ""
    verdict: str = ""


@dataclass(frozen=True)
class PairedView:
    """A head-to-head comparison of two model versions on the requests both of them answered.

    The ``n_*`` fields count what one of the two compared versions logged but could not take part,
    so the output can say exactly what was left out and why. ``n_not_gold_on_both`` and
    ``n_label_conflict`` count pairs; the others count records.
    """

    baseline_label: str
    current_label: str
    questions: tuple[PairedQuestion, ...] = ()
    n_unkeyed: int = 0
    n_duplicate: int = 0
    n_one_sided: int = 0
    n_not_gold_on_both: int = 0
    n_label_conflict: int = 0
    n_score: int = 0
    min_pairs: int = MIN_PAIRS
    alpha: float = DEFAULT_ALPHA
    note: str = ""


@dataclass(frozen=True)
class _Candidate:
    """One action's recommendation for one slice, with the cost that ranked it."""

    action: str
    threshold: float
    auto_rate: float
    cost_per_case: float


def split_models(records: Sequence[DecisionRecord]) -> tuple[DriftSlice, ...]:
    """One slice per model value, ordered by each model's first-seen timestamp.

    This is the coarse view a report charts: each population that served, measured on its own.
    Every slice is measured over one edge set pooled from the whole input, so two of these slices
    can be compared directly. A model whose gold records cannot be measured (no labels, or only
    ``score`` records) is kept with ``n=0`` and a NaN ECE; no intervals are computed here.
    """
    grouped = _group_by_model(records)
    if not grouped:
        return ()
    edges = _shared_edges(records)
    slices: list[DriftSlice] = []
    for model in sorted(grouped, key=lambda name: (_first_seen(grouped[name]), name)):
        measured, _ = _build_slice(
            grouped[model],
            label=model,
            model=model,
            edges=edges,
            alpha=DEFAULT_ALPHA,
            n_boot=0,
            seed=DEFAULT_SEED,
        )
        slices.append(measured)
    return tuple(slices)


def split_by_period(
    records: Sequence[DecisionRecord], *, period: PeriodCode = "W"
) -> tuple[DriftSlice, ...]:
    """One slice per calendar period, ordered by period start.

    ``period='W'`` groups by ISO week (``2026-W38``, Monday-based and keyed on the ISO year), and
    ``period='M'`` by calendar month (``2026-09``). A slice's ``model`` is the model serving at the
    end of that period, which is what a timeline shows when a release lands mid-period. All slices
    share one edge set pooled from the whole input.
    """
    groups = _period_groups(records, period)
    if not groups:
        return ()
    edges = _shared_edges(records)
    slices: list[DriftSlice] = []
    for key, group in groups:
        measured, _ = _build_slice(
            group,
            label=key,
            model=_serving_model(group),
            edges=edges,
            alpha=DEFAULT_ALPHA,
            n_boot=0,
            seed=DEFAULT_SEED,
        )
        slices.append(measured)
    return tuple(slices)


def detect_model_changes(records: Sequence[DecisionRecord]) -> tuple[ModelChange, ...]:
    """Record every point where the serving ``model`` changed.

    Records are ordered by timestamp, read in UTC, and the input order breaks ties, so a log that
    already arrives in order is reported as-is. A change is recorded at the first timestamp whose
    model differs from the previous record's, and labelled ``previous -> new``. The first record
    of a log is not a change: it is the only model that has served so far.
    """
    ordered = sorted(records, key=_stamp)
    if not ordered:
        return ()
    changes: list[ModelChange] = []
    previous = ordered[0].model
    for record in ordered[1:]:
        if record.model == previous:
            continue
        changes.append(
            ModelChange(
                model=record.model,
                at=_stamp(record).isoformat(),
                label=f"{previous} -> {record.model}",
            )
        )
        previous = record.model
    return tuple(changes)


def compare(
    records: Sequence[DecisionRecord],
    *,
    baseline: str | None = None,
    current: str | None = None,
    min_slice: int = DEFAULT_MIN_SLICE,
    by_period: PeriodCode | None = None,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = DEFAULT_N_BOOT,
    seed: int = DEFAULT_SEED,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
) -> DriftView:
    """Compare two populations of the same decision: a model change, or a period split.

    ``baseline`` and ``current`` select the compared populations — model values in a model
    comparison, period labels such as ``2026-W38`` in a period comparison. With neither given, the
    two most recent model versions by first-seen timestamp are used and the newest is ``current``;
    ties break by model name so the choice is deterministic. When no model is newer than an
    explicitly named ``baseline``, ``current`` falls back to that baseline and every movement is
    zero.

    Slices are emitted at the granularity the comparison needs. A model comparison gives every
    compared question its own pair of slices — all baseline-model slices in question order, then
    all current-model slices — so a movement can be attributed to a question rather than to a
    change in the mix of questions. A period comparison gives one slice per included period in
    start order.

    ``min_slice``, ``n_bins`` and ``equal_width`` apply to the whole comparison: slices below
    ``min_slice`` measurable gold records on either side of a change (or in a period) are omitted
    and named in the resulting note, and one edge set built with the caller's binning is used for
    every slice.

    With fewer than two model values and no ``by_period``, the result is an empty
    :class:`~jeval.report.model.DriftView` whose note says what is missing: drift needs two model
    versions or a period split.
    """
    if min_slice < 0:
        raise ValueError(f"min_slice must be >= 0, got {min_slice}")
    changes = detect_model_changes(records)
    if by_period is None:
        return _compare_models(
            records,
            baseline=baseline,
            current=current,
            min_slice=min_slice,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
            n_bins=n_bins,
            equal_width=equal_width,
            changes=changes,
        )
    return _compare_periods(
        records,
        baseline=baseline,
        current=current,
        period=by_period,
        min_slice=min_slice,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
        n_bins=n_bins,
        equal_width=equal_width,
        changes=changes,
    )


def compare_paired(
    records: Sequence[DecisionRecord],
    *,
    baseline: str | None = None,
    current: str | None = None,
    min_pairs: int = MIN_PAIRS,
    alpha: float = DEFAULT_ALPHA,
    n_boot: int = calibration.DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = DEFAULT_SEED,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
) -> PairedView:
    """Compare two model versions on the same requests, pair by pair.

    The two versions are resolved exactly as :func:`compare` resolves them. A **pair** is one record
    from each version with the same ``(source_key, question_key)``: shadow traffic logs both
    versions on one request, and pairing them takes the difficulty of each request out of the
    difference. The rules, each counted in the result so nothing disappears silently:

    - ``score`` records are measured on another scale and never enter a pair (``n_score``).
    - A record without ``source_key`` cannot be matched to anything (``n_unkeyed``).
    - A key logged more than once by the same version is ambiguous — there is no telling which
      answer was the one served — so every record at that key on that side is refused
      (``n_duplicate``). Picking the first or the last would be a silent choice.
    - A key only one version logged has no partner (``n_one_sided``).
    - Both sides of a pair need a gold label (``n_not_gold_on_both`` otherwise), and the two labels
      must agree: a request labeled with two different answers has no single truth for both
      versions to be judged against (``n_label_conflict``).

    Each question with at least ``min_pairs`` pairs gets a :class:`calibration.PairedComparison`
    measured over one edge set pooled from both sides of its pairs, the same rule :func:`compare`
    follows; below that the question is refused with no numbers at all.
    """
    if min_pairs < 1:
        raise ValueError(f"min_pairs must be >= 1, got {min_pairs}")
    grouped = _group_by_model(records)
    if len(grouped) < 2:
        listed = ", ".join(sorted(grouped)) or "none"
        return PairedView(
            baseline_label="",
            current_label="",
            min_pairs=min_pairs,
            alpha=alpha,
            note=(
                "a paired comparison needs two model versions: found "
                f"{len(grouped)} model value(s) ({listed})."
            ),
        )
    baseline_model, current_model = _select_models(grouped, baseline=baseline, current=current)
    if baseline_model == current_model:
        return PairedView(
            baseline_label=baseline_model,
            current_label=current_model,
            min_pairs=min_pairs,
            alpha=alpha,
            note=(
                "nothing to compare: the baseline and the current version are both "
                f"{current_model}."
            ),
        )

    n_score = n_unkeyed = n_duplicate = 0
    sides: list[dict[tuple[str, str], DecisionRecord]] = []
    questions: set[str] = set()
    for model in (baseline_model, current_model):
        keyed: dict[tuple[str, str], list[DecisionRecord]] = {}
        for record in grouped[model]:
            if record.question_type == "score":
                n_score += 1
            elif record.source_key is None:
                n_unkeyed += 1
            else:
                keyed.setdefault((record.source_key, record.question_key), []).append(record)
                questions.add(record.question_key)
        single: dict[tuple[str, str], DecisionRecord] = {}
        for key, group in keyed.items():
            if len(group) > 1:
                n_duplicate += len(group)
            else:
                single[key] = group[0]
        sides.append(single)
    before_side, after_side = sides
    n_one_sided = len(before_side.keys() ^ after_side.keys())

    n_not_gold = n_conflict = 0
    pairs: dict[str, list[tuple[DecisionRecord, DecisionRecord]]] = {key: [] for key in questions}
    for key in sorted(before_side.keys() & after_side.keys()):
        before, after = before_side[key], after_side[key]
        if not (before.is_gold and after.is_gold):
            n_not_gold += 1
        elif before.label != after.label:
            n_conflict += 1
        else:
            pairs[key[1]].append((before, after))

    measured: list[PairedQuestion] = []
    for question in sorted(pairs):
        matched = pairs[question]
        if len(matched) < min_pairs:
            measured.append(
                PairedQuestion(
                    question=question,
                    n_pairs=len(matched),
                    refusal=(
                        f"{len(matched)} gold-labeled pair(s), fewer than the {min_pairs} a paired "
                        "comparison needs; no numbers are reported"
                    ),
                )
            )
            continue
        before_conf, before_hit = _points([before for before, _ in matched])
        after_conf, after_hit = _points([after for _, after in matched])
        edges = _shared_edges(
            [record for pair in matched for record in pair],
            n_bins=n_bins,
            equal_width=equal_width,
            alpha=alpha,
        )
        result = calibration.paired_bootstrap(
            np.asarray(before_conf, dtype=float),
            np.asarray(before_hit, dtype=bool),
            np.asarray(after_conf, dtype=float),
            np.asarray(after_hit, dtype=bool),
            edges,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
        measured.append(
            PairedQuestion(
                question=question,
                n_pairs=len(matched),
                comparison=result,
                verdict=_paired_verdict(result, alpha),
            )
        )
    return PairedView(
        baseline_label=baseline_model,
        current_label=current_model,
        questions=tuple(measured),
        n_unkeyed=n_unkeyed,
        n_duplicate=n_duplicate,
        n_one_sided=n_one_sided,
        n_not_gold_on_both=n_not_gold,
        n_label_conflict=n_conflict,
        n_score=n_score,
        min_pairs=min_pairs,
        alpha=alpha,
    )


def format_paired_lines(view: PairedView) -> list[str]:
    """The paired block as plain lines: per question the pairs, three differences and a verdict.

    Every difference is ``current - baseline`` with its bootstrap interval, so a positive ECE or
    Brier difference is a worse current version and a positive accuracy difference a better one.
    A refused question prints why and no number. The last line counts every record of the two
    versions that could not be paired, by reason.
    """
    if not view.baseline_label or view.baseline_label == view.current_label:
        return [f"paired: {view.note}"] if view.note else []
    level = f"{1.0 - view.alpha:.0%}"
    lines = [
        f"paired: {view.baseline_label} -> {view.current_label} "
        "(the same requests, matched by source_key and question)"
    ]
    if not view.questions:
        lines.append("  no question has a record from both versions at the same source_key")
    for question in view.questions:
        result = question.comparison
        if result is None:
            lines.append(f"  {question.question}: refused: {question.refusal}")
            continue
        lines.append(f"  {question.question}: {question.n_pairs:,} pairs")
        rows = (("accuracy", result.accuracy), ("ECE", result.ece), ("Brier", result.brier))
        for name, metric in rows:
            line = (
                f"    {name:<8}  {metric.baseline:.3f} -> {metric.current:.3f}  "
                f"{_delta(metric.difference)}  {level} CI {_interval(metric)}"
            )
            if name == "accuracy":
                line += (
                    f"  McNemar {_p_value(result.mcnemar_p)} (right on baseline only: "
                    f"{result.discordant_baseline_only}, on current only: "
                    f"{result.discordant_current_only})"
                )
            lines.append(line)
        lines.append(f"    verdict: {question.verdict}")
    lines.append(
        f"  not paired: {view.n_unkeyed} without source_key, {view.n_duplicate} at a key logged "
        f"twice by one version, {view.n_one_sided} with no partner, {view.n_not_gold_on_both} "
        f"pair(s) without a gold label on both sides, {view.n_label_conflict} with conflicting "
        f"labels, {view.n_score} score"
    )
    return lines


def parse_fail_on(specs: Sequence[str]) -> tuple[DriftCheck, ...]:
    """Parse ``--fail-on`` specs into checks, e.g. ``'ece-increase=0.05'``.

    An unknown key or a limit that is not a finite, non-negative number raises ``ValueError``
    naming the offending spec: a typo in a CI gate must fail the build loudly, not silently
    disable a check.
    """
    checks: list[DriftCheck] = []
    allowed = ", ".join(CHECK_KEYS)
    for spec in specs:
        key, separator, raw = spec.partition("=")
        key = key.strip()
        raw = raw.strip()
        if not separator:
            raise ValueError(f"invalid drift check {spec!r}: expected KEY=LIMIT, one of {allowed}")
        if key not in CHECK_KEYS:
            raise ValueError(f"unknown drift check {spec!r}: expected one of {allowed}")
        try:
            limit = float(raw)
        except ValueError:
            raise ValueError(
                f"invalid limit in drift check {spec!r}: {raw!r} is not a number"
            ) from None
        if not math.isfinite(limit) or limit < 0:
            raise ValueError(
                f"invalid limit in drift check {spec!r}: expected a finite number >= 0"
            )
        checks.append(DriftCheck(key=key, limit=limit))
    return tuple(checks)


def run_checks(view: DriftView, checks: Sequence[DriftCheck]) -> tuple[DriftFailure, ...]:
    """Evaluate each check against the baseline and current side of every compared unit.

    - ``ece-increase``: fails when the ECE rose by more than its limit; ``value`` is the rise.
    - ``auto-rate-drop``: fails when the automation rate fell by more than its limit; ``value`` is
      the drop. A unit without an automation rate — no cost matrix was applied to those slices —
      is skipped, so a number nobody computed never becomes a silent pass.
    - ``ece-above``: fails when the *current* slice's ECE is above its limit, which catches a
      replacement model that is worse than the one it replaced even when the movement itself is
      small.

    A unit whose ECE is NaN cannot produce a numerical verdict and is skipped; ``compare`` already
    names those in the view note. Failures come back in check order, then unit order, and each
    detail starts with its unit's label so :func:`format_ci_block` can attribute it to a row.
    """
    units = _comparison_units(view)
    failures: list[DriftFailure] = []
    for check in checks:
        if check.key == CHECK_ECE_INCREASE:
            for unit in units:
                rise = unit.after.ece - unit.before.ece
                if not math.isfinite(rise) or rise <= check.limit:
                    continue
                failures.append(
                    DriftFailure(
                        check=check.key,
                        detail=(
                            f"{unit.label}: ECE {_ece(unit.before.ece)} -> "
                            f"{_ece(unit.after.ece)} ({_delta(rise)}), limit "
                            f"{check.limit:.3f}"
                        ),
                        value=rise,
                        limit=check.limit,
                    )
                )
        elif check.key == CHECK_AUTO_RATE_DROP:
            for unit in units:
                before_rate = unit.before.auto_rate
                after_rate = unit.after.auto_rate
                if before_rate is None or after_rate is None:
                    continue
                drop = before_rate - after_rate
                if drop <= check.limit:
                    continue
                failures.append(
                    DriftFailure(
                        check=check.key,
                        detail=(
                            f"{unit.label}: auto-rate {before_rate:.0%} -> {after_rate:.0%} "
                            f"({drop:.0%} drop), limit {check.limit:.0%}"
                        ),
                        value=drop,
                        limit=check.limit,
                    )
                )
        elif check.key == CHECK_ECE_ABOVE:
            for measured in _current_slices(view):
                if not math.isfinite(measured.ece) or measured.ece <= check.limit:
                    continue
                failures.append(
                    DriftFailure(
                        check=check.key,
                        detail=(
                            f"{measured.label}: current ECE {_ece(measured.ece)}, limit "
                            f"{check.limit:.3f}"
                        ),
                        value=measured.ece,
                        limit=check.limit,
                    )
                )
        else:
            allowed = ", ".join(CHECK_KEYS)
            raise ValueError(f"unknown drift check {check.key!r}; expected one of {allowed}")
    return tuple(failures)


def attach_thresholds(
    view: DriftView,
    records: Sequence[DecisionRecord],
    actions: Sequence[CostAction],
    *,
    steps: int = DEFAULT_SWEEP_STEPS,
    alpha: float = DEFAULT_ALPHA,
) -> DriftView:
    """Put a cost-derived threshold on every slice the cost matrix can speak for.

    A comparison on its own only says that calibration moved. The line a reviewer acts on is the
    one the money moves, so each slice is swept with the action whose ``question`` matches the
    slice's label: in a model comparison that label is the question key, so ``intent`` at the
    baseline model is swept over the baseline model's own ``intent`` records and set against the
    same question at the current model. A period comparison labels its slices by period instead,
    so no action matches those and they keep ``None`` — a threshold attributed to the wrong
    population is worse than no threshold.

    Where several actions share a slice's question, the cheapest recommendation wins and a tie
    goes to the first action in the caller's order; the view note names the action reported.

    Nothing is invented. A slice that cannot be swept — fewer gold-labeled records than
    ``jeval.costs.MIN_GOLD_RECORDS``, no action on its label, or no record of that question at
    that model — keeps ``threshold`` and ``auto_rate`` at ``None``, and the note names it with the
    count and the reason, so the CI block can say why instead of printing a number. The caller's
    own note is preserved, and the sentence this function writes is rewritten rather than
    appended to if it is already there, so attaching twice cannot stack two claims.

    With no actions at all there is no cost matrix, and the view comes back untouched — note
    included — so the block keeps its "no cost matrix was applied" wording instead of claiming a
    sweep that never ran.

    ``steps`` is the sweep grid and ``alpha`` the interval level; both are validated here exactly
    as :func:`jeval.costs.sweep` validates them.
    """
    if steps < 2:
        raise ValueError(f"steps must be >= 2 to sweep a range, got {steps}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not actions:
        # No cost matrix is the caller's way of saying "measure only what moved". The view comes
        # back untouched, note included, so the block keeps its "no cost matrix" wording.
        return view

    # The costs engine is owned elsewhere and is imported, never edited, here. The import is local
    # so that a call is what needs it, not every use of drift detection.
    from jeval.costs import MIN_GOLD_RECORDS, sweep

    base_note = view.note.partition(COST_MATRIX_NOTE)[0].strip()
    if not view.slices:
        sentence = f"{COST_MATRIX_NOTE}: the comparison has no slice to sweep."
        return replace(view, note=" ".join(part for part in (base_note, sentence) if part))

    groups = _question_model_groups(records)
    updated: list[DriftSlice] = []
    withheld: list[str] = []
    ambiguous: list[str] = []
    attached = 0
    for measured in view.slices:
        matched = [action for action in actions if action.question == measured.label]
        group = groups.get((measured.label, measured.model))
        if not matched:
            withheld.append(f"{measured.label} at {measured.model} (no cost action on this label)")
            updated.append(measured)
            continue
        if group is None:
            withheld.append(
                f"{measured.label} at {measured.model} (no record of this question at this model)"
            )
            updated.append(measured)
            continue
        candidates: list[_Candidate] = []
        n_records = 0
        for action in matched:
            result = sweep(action, group, steps=steps, alpha=alpha)
            n_records = max(n_records, result.n_records)
            # A threshold without an automation rate would print as a NaN in the block, so the two
            # are kept together: both or neither.
            if not (math.isfinite(result.threshold) and math.isfinite(result.auto_rate)):
                continue
            candidates.append(
                _Candidate(
                    action=action.name,
                    threshold=result.threshold,
                    auto_rate=result.auto_rate,
                    cost_per_case=result.expected_cost_per_case,
                )
            )
        if not candidates:
            withheld.append(
                f"{measured.label} at {measured.model} ({n_records} gold-labeled record(s), "
                f"under the {MIN_GOLD_RECORDS} a sweep needs)"
            )
            updated.append(measured)
            continue
        chosen = min(candidates, key=lambda candidate: candidate.cost_per_case)
        if len(candidates) > 1:
            ambiguous.append(
                f"{measured.label} at {measured.model} matched {len(candidates)} cost actions; "
                f"the cheapest recommendation ({chosen.action}) is reported."
            )
        attached += 1
        updated.append(replace(measured, threshold=chosen.threshold, auto_rate=chosen.auto_rate))

    total = len(view.slices)
    if attached:
        headline = (
            f"{COST_MATRIX_NOTE}: cost-derived thresholds attached to {attached} of "
            f"{total} slice(s)"
        )
    else:
        headline = (
            f"{COST_MATRIX_NOTE}: no cost-derived threshold could be attached to any of "
            f"{total} slice(s)"
        )
    if withheld:
        headline += "; withheld for " + "; ".join(withheld)
    sentences = [f"{headline}.", *ambiguous]
    return replace(
        view,
        slices=tuple(updated),
        note=" ".join(part for part in (base_note, *sentences) if part),
    )


def _looks_like_period(label: str) -> bool:
    """An ISO week or month label, so the block can stop calling a month a model."""
    import re

    return re.fullmatch(r"\d{4}-(W\d{2}|\d{2})", label or "") is not None


def format_ci_block(
    view: DriftView,
    failures: Sequence[DriftFailure],
    paired: PairedView | None = None,
) -> str:
    """The block a CI job prints, as plain column-aligned text.

    No colour and no decoration, because this string is read in a CI log, pasted into a bug report
    and diffed between runs. It is, in order: a header naming the change, an indented table with
    the before/after ECE and the delta per compared unit and its PASS/FAIL verdict, the threshold
    and automation-rate movement when the slices carry one, the paired head-to-head block when one
    is given, the view's note when there is one, and the exit code the block implies. The paired
    block informs; it never changes the exit code, which only the ``--fail-on`` checks decide.
    """
    if not view.baseline_label and not view.current_label:
        return _with_exit(view.note or "no drift comparison available", failures)
    if view.baseline_label == view.current_label:
        # Comparing a unit with itself produced "model changed: new -> new" and a +0.000 delta
        # that reads as a clean bill of health for a comparison nobody made.
        return _with_exit(
            f"nothing to compare: the baseline and the current slice are both "
            f"{view.baseline_label}",
            failures,
        )
    kind = "period" if _looks_like_period(view.baseline_label) else "model"
    header = f"{kind} changed: {view.baseline_label} -> {view.current_label}"
    when = _change_date(view)
    if when:
        header += f" ({when})"
    lines = [header]
    units = _comparison_units(view)
    if units:
        lines.extend(_table_lines(view, units, failures))
    lines.extend(_threshold_lines(view, units))
    if paired is not None:
        lines.extend(format_paired_lines(paired))
    if view.note:
        lines.append(f"note: {view.note}")
    return _with_exit("\n".join(lines), failures)


def _compare_models(
    records: Sequence[DecisionRecord],
    *,
    baseline: str | None,
    current: str | None,
    min_slice: int,
    alpha: float,
    n_boot: int,
    seed: int,
    n_bins: int,
    equal_width: bool,
    changes: tuple[ModelChange, ...],
) -> DriftView:
    """Model-version comparison: one slice per compared question per model."""
    grouped = _group_by_model(records)
    if len(grouped) < 2:
        listed = ", ".join(sorted(grouped)) or "none"
        return DriftView(
            baseline_label="",
            current_label="",
            changes=changes,
            note=(
                "drift needs either two model versions or a period split: found "
                f"{len(grouped)} model value(s) ({listed})."
            ),
        )

    baseline_model, current_model = _select_models(grouped, baseline=baseline, current=current)
    questions = sorted(
        {record.question_key for record in grouped[baseline_model]}
        | {record.question_key for record in grouped[current_model]}
    )

    included: list[tuple[str, list[DecisionRecord], list[DecisionRecord]]] = []
    omitted: list[str] = []
    for key in questions:
        before = [record for record in grouped[baseline_model] if record.question_key == key]
        after = [record for record in grouped[current_model] if record.question_key == key]
        before_n = len(_participants(before))
        after_n = len(_participants(after))
        if before_n < min_slice or after_n < min_slice:
            omitted.append(f"{key} ({before_n} vs {after_n})")
            continue
        included.append((key, before, after))

    notes: list[str] = []
    if omitted:
        notes.append(
            f"omitted {len(omitted)} question(s) with fewer than min_slice={min_slice} "
            "gold-labeled records on either side: " + ", ".join(omitted) + "."
        )
    if not included:
        notes.append(
            "no question had enough gold-labeled records on both sides of the change to compare."
        )
        return DriftView(
            baseline_label=baseline_model,
            current_label=current_model,
            changes=changes,
            note=" ".join(notes),
        )

    pooled: list[DecisionRecord] = []
    for _, before, after in included:
        pooled.extend(before)
        pooled.extend(after)
    edges = _shared_edges(pooled, n_bins=n_bins, equal_width=equal_width, alpha=alpha)

    before_slices: list[DriftSlice] = []
    after_slices: list[DriftSlice] = []
    interval_bits: list[str] = []
    for key, before, after in included:
        before_slice, before_interval = _build_slice(
            before,
            label=key,
            model=baseline_model,
            edges=edges,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        after_slice, after_interval = _build_slice(
            after,
            label=key,
            model=current_model,
            edges=edges,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        before_slices.append(before_slice)
        after_slices.append(after_slice)
        if math.isfinite(before_interval[0]) and math.isfinite(after_interval[0]):
            interval_bits.append(
                f"{key} {before_interval[0]:.3f}-{before_interval[1]:.3f} -> "
                f"{after_interval[0]:.3f}-{after_interval[1]:.3f}"
            )
    if interval_bits:
        notes.append("ECE 95% bootstrap intervals per question: " + "; ".join(interval_bits) + ".")

    return DriftView(
        baseline_label=baseline_model,
        current_label=current_model,
        slices=tuple([*before_slices, *after_slices]),
        changes=changes,
        note=" ".join(notes),
    )


def _compare_periods(
    records: Sequence[DecisionRecord],
    *,
    baseline: str | None,
    current: str | None,
    period: PeriodCode,
    min_slice: int,
    alpha: float,
    n_boot: int,
    seed: int,
    n_bins: int,
    equal_width: bool,
    changes: tuple[ModelChange, ...],
) -> DriftView:
    """Period comparison: one slice per included period, ordered by period start."""
    groups = _period_groups(records, period)
    if not groups:
        return DriftView(
            baseline_label="",
            current_label="",
            changes=changes,
            note=f"no records fall into a {period!r} period, so there is nothing to compare.",
        )

    included: list[tuple[str, list[DecisionRecord]]] = []
    omitted: list[str] = []
    for key, group in groups:
        count = len(_participants(group))
        if count < min_slice:
            omitted.append(f"{key} ({count})")
            continue
        included.append((key, group))

    notes: list[str] = []
    if omitted:
        notes.append(
            f"omitted {len(omitted)} period(s) with fewer than min_slice={min_slice} "
            "gold-labeled records: " + ", ".join(omitted) + "."
        )
    if len(included) < 2:
        notes.append(
            f"period grouping by {period!r} produced fewer than two comparable periods; "
            "drift needs at least two."
        )
        return DriftView(
            baseline_label="",
            current_label="",
            changes=changes,
            note=" ".join(notes),
        )

    labels = [key for key, _ in included]
    for label in (baseline, current):
        if label is not None and label not in labels:
            raise ValueError(f"unknown period {label!r}; comparable periods: {', '.join(labels)}")
    baseline_label = baseline if baseline is not None else labels[0]
    current_label = current if current is not None else labels[-1]

    pooled: list[DecisionRecord] = []
    for _, group in included:
        pooled.extend(group)
    edges = _shared_edges(pooled, n_bins=n_bins, equal_width=equal_width, alpha=alpha)

    slices: list[DriftSlice] = []
    interval_bits: list[str] = []
    for key, group in included:
        measured, interval = _build_slice(
            group,
            label=key,
            model=_serving_model(group),
            edges=edges,
            alpha=alpha,
            n_boot=n_boot,
            seed=seed,
        )
        slices.append(measured)
        if math.isfinite(interval[0]):
            interval_bits.append(f"{key} {interval[0]:.3f}-{interval[1]:.3f}")
    if interval_bits:
        notes.append("ECE 95% bootstrap intervals per period: " + "; ".join(interval_bits) + ".")

    return DriftView(
        baseline_label=baseline_label,
        current_label=current_label,
        slices=tuple(slices),
        changes=changes,
        note=" ".join(notes),
    )


def _select_models(
    grouped: dict[str, list[DecisionRecord]], *, baseline: str | None, current: str | None
) -> tuple[str, str]:
    """Resolve the compared model values, defaulting to the two most recent by first-seen."""
    order = sorted(grouped, key=lambda name: (_first_seen(grouped[name]), name))
    for label in (baseline, current):
        if label is not None and label not in grouped:
            raise ValueError(f"unknown model {label!r}; known models: {', '.join(order)}")
    current_model = current if current is not None else order[-1]
    if baseline is not None:
        return baseline, current_model
    rank = {name: (_first_seen(grouped[name]), name) for name in order}
    older = [name for name in order if rank[name] < rank[current_model]]
    return (older[-1] if older else current_model), current_model


def _group_by_model(records: Sequence[DecisionRecord]) -> dict[str, list[DecisionRecord]]:
    grouped: dict[str, list[DecisionRecord]] = {}
    for record in records:
        grouped.setdefault(record.model, []).append(record)
    return grouped


def _question_model_groups(
    records: Sequence[DecisionRecord],
) -> dict[tuple[str, str], list[DecisionRecord]]:
    """Records keyed by ``(question, model)``, which is what a model-comparison slice is keyed by.

    That pair is exactly what ``compare`` gives a model-comparison slice, so a cost sweep runs over
    the records the slice was measured on and no others: pooling the question across models would
    report the same threshold for a model that degraded and one that did not.
    """
    groups: dict[tuple[str, str], list[DecisionRecord]] = {}
    for record in records:
        groups.setdefault((record.question_key, record.model), []).append(record)
    return groups


def _period_groups(
    records: Sequence[DecisionRecord], period: PeriodCode
) -> list[tuple[str, list[DecisionRecord]]]:
    """Records grouped by period key, ordered by period start."""
    if period not in PERIOD_CODES:
        raise ValueError(f"unknown period {period!r}; expected one of {', '.join(PERIOD_CODES)}")
    grouped: dict[str, list[DecisionRecord]] = {}
    starts: dict[str, date] = {}
    for record in records:
        key, start = _period_of(_stamp(record), period)
        grouped.setdefault(key, []).append(record)
        starts[key] = min(start, starts.get(key, start))
    return [(key, grouped[key]) for key in sorted(grouped, key=lambda name: starts[name])]


def _period_of(ts: datetime, period: PeriodCode) -> tuple[str, date]:
    """Period key and its start date for a timestamp (ISO week or calendar month)."""
    if period == "W":
        iso = ts.isocalendar()
        return f"{iso.year}-W{iso.week:02d}", date.fromisocalendar(iso.year, iso.week, 1)
    return f"{ts.year}-{ts.month:02d}", date(ts.year, ts.month, 1)


def _participants(records: Sequence[DecisionRecord]) -> list[DecisionRecord]:
    """The records of a group that can take part in a binary calibration measurement.

    Gold labels only: agreement with another model is not accuracy. And ``score`` records only:
    they are excluded because they are measured on a different scale, exactly as they are in
    ``jeval.evaluate``.
    """
    return [
        record for record in records if record.is_gold and record.calibration_point() is not None
    ]


def _points(records: Sequence[DecisionRecord]) -> tuple[list[float], list[bool]]:
    confidences: list[float] = []
    correct: list[bool] = []
    for record in records:
        point = record.calibration_point()
        if point is None:
            continue
        confidences.append(point[0])
        correct.append(point[1])
    return confidences, correct


def _shared_edges(
    records: Sequence[DecisionRecord],
    *,
    n_bins: int = DEFAULT_N_BINS,
    equal_width: bool = False,
    alpha: float = DEFAULT_ALPHA,
) -> list[float]:
    """The one edge set every slice in a comparison is measured over.

    The edges come from ``calibration.compute_calibration`` run once on the pooled population of
    the comparison — the same construction rule, the same automatic bin reduction, the caller's
    binning — and are then reused for every slice. The bootstrap is not run here because it cannot
    move the edges; the per-slice intervals are computed where they are reported.
    """
    confidences, correct = _points(records)
    if not confidences:
        return []
    metrics = calibration.compute_calibration(
        confidences,
        correct,
        n_bins=n_bins,
        equal_width=equal_width,
        alpha=alpha,
        n_boot=0,
    )
    bins = metrics.bins
    if not bins:
        return []
    return [bins[0].lo, *(calibration_bin.hi for calibration_bin in bins)]


def _build_slice(
    group: Sequence[DecisionRecord],
    *,
    label: str,
    model: str,
    edges: Sequence[float],
    alpha: float,
    n_boot: int,
    seed: int,
) -> tuple[DriftSlice, tuple[float, float]]:
    """Measure one slice over the comparison's shared edges, with its bootstrap interval.

    ``n`` counts the records the ECE is actually computed from, so a slice that looks large
    because it is full of silver labels or ``score`` records cannot lie with its sample size.
    ``start``/``end`` still describe the whole group, so the window reads as the span the slice
    covers even when nothing in it can be measured.
    """
    measured = _participants(group)
    confidences, correct = _points(measured)
    n = len(confidences)
    if n and len(edges) >= 2:
        confidence_array = np.asarray(confidences, dtype=float)
        correct_array = np.asarray(correct, dtype=bool)
        bins = calibration.bins_from_edges(confidence_array, correct_array, list(edges), alpha)
        ece = calibration.expected_calibration_error(bins, n)
        interval = calibration.bootstrap_ece_ci(
            confidence_array,
            correct_array,
            list(edges),
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )
    else:
        ece = float("nan")
        interval = (float("nan"), float("nan"))
    stamps = [_stamp(record) for record in group]
    return (
        DriftSlice(
            label=label,
            model=model,
            start=min(stamps).isoformat() if stamps else "",
            end=max(stamps).isoformat() if stamps else "",
            n=n,
            ece=ece,
        ),
        interval,
    )


def _stamp(record: DecisionRecord) -> datetime:
    """A record's timestamp in UTC; a naive timestamp is read as UTC, as the schema implies."""
    ts = record.ts
    return ts.astimezone(timezone.utc) if ts.tzinfo is not None else ts.replace(tzinfo=timezone.utc)


def _first_seen(group: Sequence[DecisionRecord]) -> datetime:
    return min(_stamp(record) for record in group)


def _serving_model(group: Sequence[DecisionRecord]) -> str:
    """The model serving at the end of a period, which is what a timeline shows."""
    return max(group, key=_stamp).model


def _is_model_comparison(view: DriftView) -> bool:
    """A model comparison labels its slices with model values; a period one does not."""
    if not view.baseline_label or not view.current_label:
        return False
    models = {measured.model for measured in view.slices}
    return view.baseline_label in models and view.current_label in models


def _comparison_units(view: DriftView) -> tuple[_Unit, ...]:
    """Pair the slices of a view into the units the checks and the table read.

    A model comparison pairs a question at the baseline model with the same question at the
    current model. A period comparison pairs consecutive periods across the selected range, so the
    table reads as a trend. A unit's label is the label of its later slice, and every failure
    detail starts with it — that is how the CI block attributes a FAIL to a row.
    """
    if _is_model_comparison(view):
        after_by_label = {
            measured.label: measured
            for measured in view.slices
            if measured.model == view.current_label
        }
        return tuple(
            _Unit(label=measured.label, before=measured, after=after_by_label[measured.label])
            for measured in view.slices
            if measured.model == view.baseline_label and measured.label in after_by_label
        )
    labels = [measured.label for measured in view.slices]
    if view.baseline_label not in labels or view.current_label not in labels:
        return ()
    start = labels.index(view.baseline_label)
    end = labels.index(view.current_label)
    window = view.slices[start : end + 1]
    return tuple(
        _Unit(label=window[index].label, before=window[index - 1], after=window[index])
        for index in range(1, len(window))
    )


def _current_slices(view: DriftView) -> tuple[DriftSlice, ...]:
    """The slices on the current side of the comparison, which ``ece-above`` is judged on."""
    if _is_model_comparison(view):
        return tuple(measured for measured in view.slices if measured.model == view.current_label)
    return tuple(measured for measured in view.slices if measured.label == view.current_label)


def _table_lines(
    view: DriftView, units: Sequence[_Unit], failures: Sequence[DriftFailure]
) -> list[str]:
    """The indented before/after table, column-aligned on its widest cell."""
    failed = _failed_labels(failures)
    first_column = "question" if _is_model_comparison(view) else "period"
    label_width = max(len(first_column), *(len(unit.label) for unit in units))
    before_width = max(len("ECE before"), *(len(_ece(unit.before.ece)) for unit in units))
    after_width = max(len("ECE after"), *(len(_ece(unit.after.ece)) for unit in units))
    delta_width = max(
        len("delta"), *(len(_delta(unit.after.ece - unit.before.ece)) for unit in units)
    )
    lines = [
        "  "
        + first_column.ljust(label_width)
        + "  "
        + "ECE before".rjust(before_width)
        + "  "
        + "ECE after".rjust(after_width)
        + "  "
        + "delta".rjust(delta_width)
    ]
    for unit in units:
        delta = unit.after.ece - unit.before.ece
        lines.append(
            "  "
            + unit.label.ljust(label_width)
            + "  "
            + _ece(unit.before.ece).rjust(before_width)
            + "  "
            + _ece(unit.after.ece).rjust(after_width)
            + "  "
            + _delta(delta).rjust(delta_width)
            + "   "
            + ("FAIL" if unit.label in failed else "ok")
        )
    return lines


def _threshold_lines(view: DriftView, units: Sequence[_Unit]) -> list[str]:
    """Recommended-threshold movement, when the compared slices carry one.

    Costs are the caller's: a slice carries a threshold only if :func:`attach_thresholds` measured
    one for it (``DriftSlice.threshold`` and ``DriftSlice.auto_rate``). A unit with a threshold on
    one side only prints nothing, because a movement needs both numbers. When no unit has one, the
    block says so rather than printing a threshold nobody computed — and it says which of the two
    reasons applies: no cost matrix at all, or one whose sweep withheld every threshold, in which
    case the view note names the slices and why.
    """
    lines: list[str] = []
    for unit in units:
        before_threshold = unit.before.threshold
        after_threshold = unit.after.threshold
        before_rate = unit.before.auto_rate
        after_rate = unit.after.auto_rate
        if (
            before_threshold is None
            or after_threshold is None
            or before_rate is None
            or after_rate is None
        ):
            continue
        lines.append(
            f"recommended threshold ({unit.label}): {before_threshold:.2f} -> {after_threshold:.2f}"
        )
        lines.append(
            f"  at the current {before_threshold:.2f}: "
            f"auto-rate {before_rate:.0%} -> {after_rate:.0%}"
        )
    if not lines:
        if COST_MATRIX_NOTE in view.note:
            lines.append(
                "recommended threshold: not available "
                "(a cost matrix was applied but the sweep produced no threshold)"
            )
        else:
            lines.append(
                "recommended threshold: not available "
                "(no cost matrix was applied to this comparison)"
            )
    return lines


def _failed_labels(failures: Sequence[DriftFailure]) -> frozenset[str]:
    """The unit labels the given failures belong to; a detail always starts with its label."""
    labels = (failure.detail.partition(":")[0].strip() for failure in failures)
    return frozenset(label for label in labels if label)


def _change_date(view: DriftView) -> str:
    """Short date of the change into the current population, for the block's first line."""
    for change in reversed(view.changes):
        if change.model == view.current_label:
            return _short_date(change.at)
    units = _comparison_units(view)
    return _short_date(units[-1].after.end) if units else ""


def _short_date(value: str) -> str:
    """``Sep 18`` for an ISO timestamp; the input itself when it cannot be read as one."""
    try:
        stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    if stamp.tzinfo is not None:
        stamp = stamp.astimezone(timezone.utc)
    return f"{stamp:%b %d}"


def _ece(value: float) -> str:
    """Three decimals, or ``nan`` — never a fake zero for a slice with no measurable records."""
    return f"{value:.3f}"


def _delta(value: float) -> str:
    return f"{value:+.3f}" if math.isfinite(value) else "n/a"


def _paired_verdict(result: calibration.PairedComparison, alpha: float) -> str:
    """One line in plain words that claims only what the sample resolves.

    Accuracy is judged by the exact McNemar test on the discordant pairs, calibration by whether
    the ECE difference interval excludes zero. A difference inside the noise is called unresolved,
    never "no difference", which a sample cannot show.
    """
    p_text = _p_value(result.mcnemar_p)
    if result.mcnemar_p < alpha:
        word = "less" if result.accuracy.difference < 0 else "more"
        accuracy = f"current is {word} accurate beyond noise (McNemar {p_text})"
    else:
        accuracy = f"no accuracy difference the sample can resolve (McNemar {p_text})"
    ece = result.ece
    if math.isfinite(ece.ci_low) and (ece.ci_low > 0.0 or ece.ci_high < 0.0):
        word = "worse" if ece.difference > 0 else "better"
        calibrated = (
            f"current is {word} calibrated beyond noise "
            f"(ECE {_delta(ece.difference)}, interval excludes 0)"
        )
    else:
        calibrated = "no calibration difference the sample can resolve"
    return f"{accuracy}; {calibrated}"


def _p_value(value: float) -> str:
    return "p<0.001" if value < 0.001 else f"p={value:.3f}"


def _interval(metric: calibration.PairedDifference) -> str:
    """``low to high``, or ``n/a`` when every resample agreed and there is no interval to state."""
    if not (math.isfinite(metric.ci_low) and math.isfinite(metric.ci_high)):
        return "n/a"
    return f"{metric.ci_low:+.3f} to {metric.ci_high:+.3f}"


def _with_exit(block: str, failures: Sequence[DriftFailure]) -> str:
    """Close the block with the exit code it implies."""
    return block + ("\nexit 1" if failures else "\nexit 0")
