"""The predefined performance suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.performance import PerformanceReport, ConfusionMatrixReport, RocReport, NaiveModelComparison, \
    CalibrationMetric


__all__ = ['ClassificationCheckSuite', 'PerformanceCheckSuite', 'GenericPerformanceCheckSuite', 'RegressionCheckSuite']


ClassificationCheckSuite = CheckSuite(
    'Classification Suite',
    ConfusionMatrixReport(),
    RocReport(),
    CalibrationMetric(),
)


RegressionCheckSuite = CheckSuite(
    'Regression Suite',
)


GenericPerformanceCheckSuite = CheckSuite(
    'Generic Performance Suite',
    PerformanceReport(),
    NaiveModelComparison()
)


PerformanceCheckSuite = CheckSuite(
    'Performance Suite',
    GenericPerformanceCheckSuite,
    ClassificationCheckSuite
)
