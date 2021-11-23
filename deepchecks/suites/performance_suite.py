"""The predefined performance suite module."""
from deepchecks import CheckSuite
from deepchecks.checks import TrustScoreComparison
from deepchecks.checks.performance import PerformanceReport, ConfusionMatrixReport, RocReport, NaiveModelComparison, \
    CalibrationMetric


__all__ = ['ClassificationCheckSuite', 'PerformanceCheckSuite', 'GenericPerformanceCheckSuite', 'RegressionCheckSuite']


ClassificationCheckSuite = CheckSuite(
    'Classification Suite',
    ConfusionMatrixReport(),
    RocReport().add_condition_auc_not_less_than(),
    CalibrationMetric(),
    TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than()
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
