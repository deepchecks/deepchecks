"""The predefined performance suite module."""
from deepchecks import Suite
from deepchecks.checks import TrustScoreComparison
from deepchecks.checks.performance import (
    PerformanceReport,
    ConfusionMatrixReport,
    RocReport,
    SimpleModelComparison,
    CalibrationMetric

)

__all__ = [
    'classification_suite',
    'regression_suite',
    'generic_performance_suite',
    'regression_suite',
    'performance_suite'
]


def classification_suite() -> Suite:
    """Create 'Classification Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the classification model.
    """
    return Suite(
        'Classification Suite',
        ConfusionMatrixReport(),
        RocReport().add_condition_auc_not_less_than(),
        CalibrationMetric(),
        TrustScoreComparison().add_condition_mean_score_percent_decline_not_greater_than()
    )


def regression_suite() -> Suite:
    """Create 'Regression Suite'.

    The suite runs a set of checks that are meant to measure and detect performance
    abnormality of the regression model.
    """
    # TODO: This suite is here as a placeholder for future regression-specific checks
    return Suite('Regression Suite')


def generic_performance_suite() -> Suite:
    """Create 'Generic Performance Suite'.

    The suite runs a set of checks that are meant to measure and detect performance abnormality in any model type.
    """
    return Suite(
        'Generic Performance Suite',
        PerformanceReport(),
        SimpleModelComparison().add_condition_ratio_not_less_than()
    )


def performance_suite() -> Suite:
    """Create 'Performance Suite'.

    The suite runs all checks that are meant to measure and detect performance abnormality in a model.

    The suite includes checks from 'Generic Performance Suite', 'Classification Suite'
    and 'Regression Suite'.
    """
    return Suite(
        'Performance Suite',
        generic_performance_suite(),
        classification_suite()
    )
