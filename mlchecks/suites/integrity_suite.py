"""The predefined Integrity suite module."""

from mlchecks import CheckSuite
from mlchecks.checks import IsSingleValue, MixedNulls, MixedTypes, StringMismatch

__all__ = ['IntegrityCheckSuite']

IntegrityCheckSuite = CheckSuite('Integrity Suite',
    IsSingleValue(),
    MixedNulls(),
    MixedTypes(),
    StringMismatch()
)
