"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.overfit import TrainTestDifferenceOverfit, BoostingOverfit


__all__ = ['OverfitCheckSuite']


OverfitCheckSuite = CheckSuite(
    'Overfit Suite',
    TrainTestDifferenceOverfit(),
    BoostingOverfit()
)
