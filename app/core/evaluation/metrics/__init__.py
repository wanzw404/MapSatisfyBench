"""Deterministic metric calculators.

Each sub-module exports a pure function ``compute_<metric>(...)`` that maps
the already-judged verdict + ground truth into a float in [0, 1].
The calculator in ``calculator.py`` simply wires them together; there is
no weighted aggregation at the per-case level — a case is reported as
its eight independent metrics.
"""

from .calculator import METRIC_NAMES, MetricCalculator
from .cei import compute_cei
from .eff import compute_eff
from .es import compute_ses
from .ecr import compute_ecr
from .ifs import compute_ifs
from .iisr import compute_iisr
from .ts import compute_ts

__all__ = [
    "MetricCalculator",
    "METRIC_NAMES",
    "compute_ecr",
    "compute_ts",
    "compute_ifs",
    "compute_iisr",
    "compute_eff",
    "compute_ses",
    "compute_cei",
]
