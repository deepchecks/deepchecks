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
    CategoryMismatchTrainTest,
    NewLabelTrainTest,
    RareFormatDetection,
    DominantFrequencyChange,
    StringLengthOutOfBounds,
)

__all__ = ['SingleDatasetIntegrityCheckSuite', 'ComparativeIntegrityCheckSuite', 'IntegrityCheckSuite']

SingleDatasetIntegrityCheckSuite = CheckSuite(
    'Single Dataset Integrity Suite',
    IsSingleValue(),
    MixedNulls().add_condition_different_nulls_not_more_than(),
    MixedTypes().add_condition_rare_type_ratio_not_less_than(),
    StringMismatch().add_condition_no_variants(),
    DataDuplicates().add_condition_duplicates_not_greater_than(),
    RareFormatDetection().add_condition_ratio_of_rare_formats_not_greater_than(),
    StringLengthOutOfBounds(),
    SpecialCharacters()
)


ComparativeIntegrityCheckSuite = CheckSuite(
    'Comparative Integrity Suite',
    StringMismatchComparison().add_condition_no_new_variants(),
    CategoryMismatchTrainTest().add_condition_new_categories_not_greater_than(),
    DominantFrequencyChange().add_condition_p_value_not_less_than(),
    NewLabelTrainTest().add_condition_new_labels_not_greater_than()
)

IntegrityCheckSuite = CheckSuite(
    'Integrity Suite',
    SingleDatasetIntegrityCheckSuite,
    ComparativeIntegrityCheckSuite
)
