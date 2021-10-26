"""The predefined performance suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.performance import PerformanceReport, ConfusionMatrixReport, RocReport


__all__ = ['PerformanceCheckSuite']


PerformanceCheckSuite = CheckSuite(
    'Performance Suite',
    PerformanceReport(),
    ConfusionMatrixReport(),
    RocReport()
)
