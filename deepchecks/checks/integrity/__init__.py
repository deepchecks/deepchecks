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
"""Module contains all data integrity checks."""
from .mixed_nulls import MixedNulls
from .string_mismatch import StringMismatch
from .mixed_data_types import MixedDataTypes
from .is_single_value import IsSingleValue
from .special_chars import SpecialCharacters
from .string_length_out_of_bounds import StringLengthOutOfBounds
from .string_mismatch_comparison import StringMismatchComparison
from .dominant_frequency_change import DominantFrequencyChange
from .data_duplicates import DataDuplicates
from .new_category import CategoryMismatchTrainTest
from .new_label import NewLabelTrainTest
from .label_ambiguity import LabelAmbiguity


__all__ = [
    'MixedNulls',
    'StringMismatch',
    'MixedDataTypes',
    'IsSingleValue',
    'SpecialCharacters',
    'StringLengthOutOfBounds',
    'StringMismatchComparison',
    'DominantFrequencyChange',
    'DataDuplicates',
    'CategoryMismatchTrainTest',
    'NewLabelTrainTest',
    'LabelAmbiguity',
]
