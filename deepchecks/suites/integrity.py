# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The predefined Integrity suite module."""
from deepchecks import Suite
from deepchecks.checks.integrity import (
    IsSingleValue,
    MixedNulls,
    MixedTypes,
    StringMismatch,
    DataDuplicates,
    SpecialCharacters,
    StringMismatchComparison,
    CategoryMismatchTrainTest,
    DominantFrequencyChange,
    StringLengthOutOfBounds,
    LabelAmbiguity
)


__all__ = [
    'single_dataset_integrity_suite',
    'comparative_integrity_suite',
    'integrity_suite'
]


def single_dataset_integrity_suite() -> Suite:
    """Create 'Single Dataset Integrity Suite'.

    The suite runs a set of checks that are meant to detect integrity issues within a single dataset.
    """
    return Suite(
        'Single Dataset Integrity Suite',
        IsSingleValue().add_condition_not_single_value(),
        MixedNulls().add_condition_different_nulls_not_more_than(),
        MixedTypes().add_condition_rare_type_ratio_not_less_than(),
        StringMismatch().add_condition_no_variants(),
        DataDuplicates().add_condition_ratio_not_greater_than(),
        StringLengthOutOfBounds().add_condition_ratio_of_outliers_not_greater_than(),
        SpecialCharacters().add_condition_ratio_of_special_characters_not_grater_than(),
        LabelAmbiguity().add_condition_ambiguous_sample_ratio_not_greater_than()
    )


def comparative_integrity_suite() -> Suite:
    """Create 'Comparative Integrity Suite'.

    The suite runs a set of checks that compare between two datasets to detect integrity issues.
    """
    return Suite(
        'Comparative Integrity Suite',
        StringMismatchComparison().add_condition_no_new_variants(),
        CategoryMismatchTrainTest().add_condition_new_categories_not_greater_than(),
        DominantFrequencyChange().add_condition_ratio_of_change_not_more_than(),
    )


def integrity_suite() -> Suite:
    """Create 'Integrity Suite'.

    The suite runs all checks intended to detect integrity issues within datasets and comparing between datasets.

    Suite includes 'Comparative Integrity Suite' and 'Single Dataset Integrity Suite'.
    """
    return Suite(
        'Integrity Suite',
        single_dataset_integrity_suite(),
        comparative_integrity_suite()
    )
