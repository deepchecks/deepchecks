"""The predefined Integrity suite module."""

from deepchecks import CheckSuite
from deepchecks.checks.integrity import (
    IsSingleValue,
    MixedNulls,
    MixedTypes,
    StringMismatch,
    DataDuplicates,
    SpecialCharacters,
    StringMismatchComparison,
    CategoryMismatchTrainValidation,
    RareFormatDetection
)

__all__ = ['SingleDatasetIntegrityCheckSuite', 'ComparativeIntegrityCheckSuite', 'IntegrityCheckSuite']

SingleDatasetIntegrityCheckSuite = CheckSuite(
    'Single Dataset Integrity Suite',
    IsSingleValue(),
    MixedNulls().add_condition_different_nulls_not_more_than(),
    MixedTypes().add_condition_rare_type_ratio_not_less_than(),
    StringMismatch().add_condition_no_variants(),
    DataDuplicates(),
    RareFormatDetection(),
    SpecialCharacters()
)


ComparativeIntegrityCheckSuite = CheckSuite(
    'Comparative Integrity Suite',
    StringMismatchComparison().add_condition_no_new_variants(),
    CategoryMismatchTrainValidation()
)

IntegrityCheckSuite = CheckSuite(
    'Integrity Suite',
    SingleDatasetIntegrityCheckSuite,
    ComparativeIntegrityCheckSuite
)
