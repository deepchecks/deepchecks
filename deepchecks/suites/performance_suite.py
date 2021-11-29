"""The predefined performance suite module."""
from deepchecks import CheckSuite
from deepchecks.checks import TrustScoreComparison
from deepchecks.checks.performance import (
    PerformanceReport,
    ConfusionMatrixReport,
    RocReport,
    NaiveModelComparison,
    CalibrationMetric,
    ClassPerformanceImbalanceCheck

)

__all__ = [
    'classification_check_suite',
    'regression_check_suite',
    'generic_performance_check_suite',
    'regression_check_suite',
    'performance_check_suite'
]


def classification_check_suite() -> CheckSuite:
    """Create 'Classification Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the classification model.
    """
    return CheckSuite(
        'Classification Suite',
        ConfusionMatrixReport(),
        RocReport().add_condition_auc_not_less_than(),
        CalibrationMetric(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than(),
        ClassPerformanceImbalanceCheck().add_condition_ratio_difference_not_greater_than()
    )


def regression_check_suite() -> CheckSuite:
    """Create 'Regression Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the regression model.
    """
    # TODO: This suite is here as a placeholder for future regression-specific checks
    return CheckSuite('Regression Suite')


def generic_performance_check_suite() -> CheckSuite:
    """Create 'Generic Performance Suite'.

    The suite runs a set of checks that are meant to measure and detect performance abnormality in any model type.
    """
    return CheckSuite(
        'Generic Performance Suite',
        PerformanceReport(),
        NaiveModelComparison().add_condition_ratio_not_less_than()
    )


def performance_check_suite() -> CheckSuite:
    """Create 'Performance Suite'.

    The suite runs all checks that are meant to measure and detect performance abnormality in a model.

    The suite includes checks from 'Generic Performance Suite', 'Classification Suite'
    and 'Regression Suite'.
    """
    return CheckSuite(
        'Performance Suite',
        generic_performance_check_suite(),
        classification_check_suite()
    )
