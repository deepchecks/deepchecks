"""The predefined performance suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.performance import PerformanceReport, ConfusionMatrixReport, RocReport, NaiveModelComparison, \
    CalibrationMetric


__all__ = ['ClassificationCheckSuite', 'PerformanceCheckSuite', 'GenericPerformanceCheckSuite', 'RegressionCheckSuite']


ClassificationCheckSuite = CheckSuite(
    'Classification Suite',
    ConfusionMatrixReport(),
    RocReport().add_condition_auc_not_less_than(),
    CalibrationMetric(),
)


# This suite is here as a placeholder for future regression-specific checks
RegressionCheckSuite = CheckSuite(
    'Regression Suite',
)


GenericPerformanceCheckSuite = CheckSuite(
    'Generic Performance Suite',
    PerformanceReport(),
    NaiveModelComparison().add_condition_ratio_not_less_than()
)


PerformanceCheckSuite = CheckSuite(
    'Performance Suite',
    GenericPerformanceCheckSuite,
    ClassificationCheckSuite
)
