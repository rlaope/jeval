"""jeval: measure the calibration of probabilistic classifiers.

jeval answers two operational questions with measurement instead of folklore:

1. When a classifier says "0.9", how often is it right?
2. Given what a mistake costs and what an escalation costs, where should the human
   hand-off line sit?

It is provider-neutral: anything that returns a probability works.
"""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

# The distribution name, which is not the import name: `jeval` on PyPI belongs to an unrelated
# project, so asking for it either raises or — worse, when that project is installed in the same
# environment — reports a version that is not ours.
DISTRIBUTION = "jeval-cli"

try:  # pragma: no cover - trivial
    __version__ = version(DISTRIBUTION)
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0+source"  # never a released version, so it cannot look like one

__all__ = ["__version__"]
