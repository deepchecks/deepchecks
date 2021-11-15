"""The predefined overfit suite module."""
from deepchecks import CheckSuite
from deepchecks.checks.overfit import TrainValidationDifferenceOverfit


__all__ = ['OverfitCheckSuite']


OverfitCheckSuite = CheckSuite(
    'Overfit Suite',
    TrainValidationDifferenceOverfit()
)
