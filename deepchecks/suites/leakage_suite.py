"""The predefined Leakage suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.leakage import (
    DataSampleLeakageReport,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    IndexTrainTestLeakage,
    SingleFeatureContributionTrainTest,
    SingleFeatureContribution
)


__all__ = [
    'index_leakage_check_suite',
    'date_leakage_check_suite',
    'data_leakage_check_suite',
    'leakage_check_suite'
]


def index_leakage_check_suite() -> CheckSuite:
    """Create 'Index Leakage Suite'.

    The suite runs a set of checks that are meant to detect problematic 
    splitting of the data between train and test, as reflected by the index column.
    """
    return CheckSuite(
        'Index Leakage Suite',
        IndexTrainTestLeakage().add_condition_ratio_not_greater_than(),
    )


def date_leakage_check_suite() -> CheckSuite:
    """Create 'Date Leakage Suite'.

    The suite runs a set of checks that tries to detect cases of problematic 
    splitting, when problematic splitting is a state in which the performance 
    on test won't represent real world performance due to it's relation in time 
    to the training data.
    """
    return CheckSuite(
        'Date Leakage Suite',
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(),
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than()
    )


def data_leakage_check_suite() -> CheckSuite:
    """Create 'Data Leakage Suite'.

    The suite runs a set of checks that are meant to detect row-wise data leakage
    from the training dataset to the test dataset, and find indications of leakage 
    by analyzing the predictive power of each feature.
    """
    return CheckSuite(
        'Data Leakage Suite',
        DataSampleLeakageReport().add_condition_duplicates_ratio_not_greater_than(),
        SingleFeatureContribution().add_condition_feature_pps_not_greater_than(),
        SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than()
    )


def leakage_check_suite() -> CheckSuite:
    """Create 'Leakage Check Suite'.

    The suite runs a set of checks that are meant to detect data
    leakage from the training dataset to the test dataset.

    The suite includes 'Data Leakage Suite', 'Date Leakage Suite', 'Index Leakage Suite'
    """
    return CheckSuite(
        'Leakage Check Suite',
        index_leakage_check_suite(),
        date_leakage_check_suite(),
        data_leakage_check_suite()
    )
