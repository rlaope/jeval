"""jeval: measure the calibration of probabilistic classifiers.

jeval answers two operational questions with measurement instead of folklore:

1. When a classifier says "0.9", how often is it right?
2. Given what a mistake costs and what an escalation costs, where should the human
   hand-off line sit?

It is provider-neutral: anything that returns a probability works.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:  # pragma: no cover - trivial
    __version__ = version("jeval")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.1.0"

__all__ = ["__version__"]
