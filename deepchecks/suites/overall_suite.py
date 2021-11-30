"""The predefined overfit suite module."""
from deepchecks.suites.methodology_suite import methodological_flaws_suite
from deepchecks.suites.distribution_suite import data_distribution_suite
from deepchecks import Suite
from deepchecks.suites import (
    integrity_suite,
    performance_suite,
    classification_suite,
    regression_suite,
    generic_performance_suite,
)


__all__ = [
    'overall_suite',
    'overall_classification_suite',
    'overall_regression_suite',
    'overall_generic_suite'
]


def overall_suite() -> Suite:
    """Create 'Overall Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return Suite(
        'Overall Suite',
        data_distribution_suite(),
        methodological_flaws_suite(),
        performance_suite(),
        integrity_suite(),
    )


def overall_classification_suite() -> Suite:
    """Create 'Overall Classification Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a classification model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return Suite(
        'Overall Classification Suite',
        data_distribution_suite(),
        methodological_flaws_suite(),
        classification_suite(),
        integrity_suite(),
    )


def overall_regression_suite() -> Suite:
    """Create 'Overall Regression Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a regression model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return Suite(
        'Overall Regression Suite',
        data_distribution_suite(),
        methodological_flaws_suite(),
        regression_suite(),
        integrity_suite(),
    )


def overall_generic_suite() -> Suite:
    """Create 'Overall Generic Suite'.

    Composition of different builtin suites that include checks that are meant
    to detect any possible issues within a model, like:
    - train/test datasets integrity issues
    - Methodological flaws such as leakage and overfit
    - Data distribution issues
    - model performance issues
    """
    return Suite(
        'Overall Generic Suite',
        data_distribution_suite(),
        methodological_flaws_suite(),
        generic_performance_suite(),
        integrity_suite(),
    )
