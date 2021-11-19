"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.overfit import TrainTestDifferenceOverfit, BoostingOverfit


__all__ = ['OverfitCheckSuite']


OverfitCheckSuite = CheckSuite(
    'Overfit Suite',
    TrainTestDifferenceOverfit() \
        .add_condition_train_test_difference_not_greater_than(0.1) \
        .add_condition_train_test_ratio_not_greater_than(0.1), # TODO: what values to use as default for conditions
    BoostingOverfit()
)
