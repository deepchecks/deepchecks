"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.suites import (
    integrity_check_suite,
    data_leakage_check_suite,
    overfit_check_suite,
    performance_check_suite,
    classification_check_suite,
    regression_check_suite,
    generic_performance_check_suite,
    leakage_check_suite
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
    - data leakage issues
    - model overfit issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Suite',
        leakage_check_suite(),
        overfit_check_suite(),
        performance_check_suite(),
        integrity_check_suite(),
    )


def overall_classification_check_suite() -> CheckSuite:
    """Create 'Overall Classification Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within classification model, like:
    - train/test datasets integrity issues
    - data leakage issues
    - model overfit issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Classification Suite',
        data_leakage_check_suite(),
        overall_check_suite(),
        classification_check_suite(),
        integrity_check_suite(),
    )


def overall_regression_check_suite() -> CheckSuite:
    """Create 'Overall Regression Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within regression model, like:
    - train/test datasets integrity issues
    - data leakage issues
    - model overfit issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Regression Suite',
        data_leakage_check_suite(),
        overfit_check_suite(),
        regression_check_suite(),
        integrity_check_suite(),
    )


def overall_generic_check_suite() -> CheckSuite:
    """Create 'Overall Generic Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within model, like:
    - train/test datasets integrity issues
    - data leakage issues
    - model overfit issues
    - model performance issues
    """
    return CheckSuite(
        'Overall Generic Suite',
        data_leakage_check_suite(),
        overfit_check_suite(),
        generic_performance_check_suite(),
        integrity_check_suite(),
    )
