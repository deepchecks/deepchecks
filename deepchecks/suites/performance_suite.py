"""The predefined performance suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.performance import (
    PerformanceReport,
    ConfusionMatrixReport,
    RocReport,
    NaiveModelComparison,
    CalibrationMetric
)


__all__ = [
    'classification_check_suite',
    'regression_check_suite',
    'generic_performance_check_suite',
    'regression_check_suite',
    'performance_check_suite'
]


def classification_check_suite() -> CheckSuite:
    """Create 'Classification Suite'."""
    return CheckSuite(
        'Classification Suite',
        ConfusionMatrixReport(),
        RocReport().add_condition_auc_not_less_than(),
        CalibrationMetric(),
    )


def regression_check_suite() -> CheckSuite:
    """Create 'Regression Suite'."""
    # TODO: This suite is here as a placeholder for future regression-specific checks
    return CheckSuite('Regression Suite',)


def generic_performance_check_suite() -> CheckSuite:
    """Create 'Generic Performance Suite'."""
    return CheckSuite(
        'Generic Performance Suite',
        PerformanceReport(),
        NaiveModelComparison().add_condition_ratio_not_less_than()
    )


def performance_check_suite() -> CheckSuite:
    """Create 'Performance Suite'."""
    return CheckSuite(
        'Performance Suite',
        generic_performance_check_suite(),
        classification_check_suite()
    )
