"""Single-file HTML reporting."""

from __future__ import annotations

from jeval.report.html import render_report, score_section, write_report
from jeval.report.svg import histogram, metric_bar, reliability_diagram

__all__ = [
    "histogram",
    "metric_bar",
    "reliability_diagram",
    "render_report",
    "score_section",
    "write_report",
]
