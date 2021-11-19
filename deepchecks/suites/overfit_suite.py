"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.overfit import TrainTestDifferenceOverfit, BoostingOverfit, UnusedFeatures


__all__ = ['OverfitCheckSuite']


OverfitCheckSuite = CheckSuite(
    'Overfit Suite',
    TrainTestDifferenceOverfit(),
    BoostingOverfit(),
    UnusedFeatures().add_condition_number_of_high_variance_unused_features_not_greater_than()
)
