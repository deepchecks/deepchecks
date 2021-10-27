"""The predefined Integrity suite module."""
from mlchecks import CheckSuite
from mlchecks.checks.integrity import (
    IsSingleValue,
    MixedNulls,
    MixedTypes,
    StringMismatch,
    DataDuplicates,
    SpecialCharacters,
    StringMismatchComparison,
    CategoryMismatchTrainValidation
)

__all__ = ['SingleDatasetIntegrityCheckSuite', 'ComparativeIntegrityCheckSuite', 'IntegrityCheckSuite']

SingleDatasetIntegrityCheckSuite = CheckSuite(
    'Single Dataset Integrity Suite',
    IsSingleValue(),
    MixedNulls(),
    MixedTypes(),
    StringMismatch(),
    DataDuplicates(),
    # RareFormatDetection(),  # Commented out until fixed
    SpecialCharacters()
)


ComparativeIntegrityCheckSuite = CheckSuite(
    'Comparative Integrity Suite',
    StringMismatchComparison(),
    CategoryMismatchTrainValidation()
)

IntegrityCheckSuite = CheckSuite(
    'Integrity Suite',
    SingleDatasetIntegrityCheckSuite,
    ComparativeIntegrityCheckSuite
)
