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
    """Create 'Index Leakage Suite'."""
    return CheckSuite(
        'Index Leakage Suite',
        IndexTrainTestLeakage().add_condition_ratio_not_greater_than(),
    )


def date_leakage_check_suite() -> CheckSuite:
    """Create 'Date Leakage Suite'."""
    return CheckSuite(
        'Date Leakage Suite',
        DateTrainTestLeakageDuplicates().add_condition_leakage_ratio_not_greater_than(),
        DateTrainTestLeakageOverlap().add_condition_leakage_ratio_not_greater_than()
    )


def data_leakage_check_suite() -> CheckSuite:
    """Create 'Data Leakage Suite'."""
    return CheckSuite(
        'Data Leakage Suite',
        DataSampleLeakageReport().add_condition_duplicates_ratio_not_greater_than(),
        SingleFeatureContribution().add_condition_feature_pps_not_greater_than(),
        SingleFeatureContributionTrainTest().add_condition_feature_pps_difference_not_greater_than()
    )


def leakage_check_suite() -> CheckSuite:
    """Create 'Leakage Check Suite'."""
    return CheckSuite(
        'Leakage Check Suite',
        index_leakage_check_suite(),
        date_leakage_check_suite,
        data_leakage_check_suite
    )
