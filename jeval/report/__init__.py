"""Single-file HTML reporting.

The report is an argument document, read top to bottom: verdict, reliability, cost, impact,
segments, drift, data quality. Modules here stay free of I/O so the whole document is a pure
function of its inputs and can be snapshot-tested.

Keep this package's ``__init__`` free of re-exports: importing ``jeval.report.model`` must not
drag in the chart layer.
"""
