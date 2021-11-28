"""The predefined overfit suite module."""
from deepchecks.suites.methodology_suite import methodological_flaws_check_suite
from deepchecks.suites.distribution_suite import data_distribution_suite
from deepchecks import CheckSuite
from deepchecks.suites import (
    integrity_check_suite,
    performance_check_suite,
    classification_check_suite,
    regression_check_suite,
    generic_performance_check_suite,
)


__all__ = [
    'overall_check_suite',
    'overall_classification_check_suite',
    'overall_regression_check_suite',
    'overall_generic_check_suite'
]


def overall_check_suite() -> CheckSuite:
    """Create 'Overall Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Suite',
        data_distribution_suite(),
        methodological_flaws_check_suite(),
        performance_check_suite(),
        integrity_check_suite(),
    )


def overall_classification_check_suite() -> CheckSuite:
    """Create 'Overall Classification Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a classification model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Classification Suite',
        data_distribution_suite(),
        methodological_flaws_check_suite(),
        classification_check_suite(),
        integrity_check_suite(),
    )


def overall_regression_check_suite() -> CheckSuite:
    """Create 'Overall Regression Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a regression model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Regression Suite',
        data_distribution_suite(),
        methodological_flaws_check_suite(),
        regression_check_suite(),
        integrity_check_suite(),
    )


def overall_generic_check_suite() -> CheckSuite:
    """Create 'Overall Generic Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Generic Suite',
        data_distribution_suite(),
        methodological_flaws_check_suite(),
        generic_performance_check_suite(),
        integrity_check_suite(),
    )
