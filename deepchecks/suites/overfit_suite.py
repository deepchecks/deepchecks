"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.overfit import TrainTestDifferenceOverfit, BoostingOverfit, UnusedFeatures


__all__ = ['overfit_check_suite']


def overfit_check_suite() -> CheckSuite:
    """Create 'Overfit Suite'."""
    return CheckSuite(
        'Overfit Suite',
        TrainTestDifferenceOverfit().add_condition_percentage_degradation_not_greater_than(),
        BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(),
        UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than()
    )
