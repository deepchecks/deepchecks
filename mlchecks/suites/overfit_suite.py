"""The predefined overfit suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.overfit import *


__all__ = ['OverfitCheckSuite']


OverfitCheckSuite = CheckSuite(
    'Overfit Suite',
    TrainValidationDifferenceOverfit()
)
