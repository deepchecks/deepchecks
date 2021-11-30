"""The predefined methodological flaws suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.methodology import (
    TrainTestSamplesMix,
    DateTrainTestLeakageDuplicates,
    DateTrainTestLeakageOverlap,
    IndexTrainTestLeakage,
    SingleFeatureContributionTrainTest,
    SingleFeatureContribution,
    TrainTestDifferenceOverfit,
    BoostingOverfit,
    UnusedFeatures,
    # ModelInferenceTimeCheck
)


__all__ = [
    'index_leakage_check_suite',
    'date_leakage_check_suite',
    'data_leakage_check_suite',
    'leakage_check_suite',
    'overfit_check_suite',
    'methodological_flaws_check_suite'
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
    splitting - a state in which the performance
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
        TrainTestSamplesMix().add_condition_duplicates_ratio_not_greater_than(),
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


def overfit_check_suite() -> CheckSuite:
    """Create 'Overfit Suite'.

    The suite runs a set of checks that are meant to detect whether
    the model was overfitted or not.
    """
    return CheckSuite(
        'Overfit Check Suite',
        TrainTestDifferenceOverfit().add_condition_percentage_degradation_not_greater_than(),
        BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(),
    )


def methodological_flaws_check_suite() -> CheckSuite:
    """Create 'Methodology Flaws Check Suite'.

    The suite runs a set of checks that are meant to detect methodological flaws in the model building process.

    The suite includes 'Leakage Check Suite', 'Overfit Check Suite', 'UnusedFeatures check'
    """
    return CheckSuite(
        'Methodological Flaws Check Suite',
        leakage_check_suite(),
        overfit_check_suite(),
        UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than(),
        # ModelInferenceTimeCheck().add_condition_inference_time_is_not_greater_than()
    )
