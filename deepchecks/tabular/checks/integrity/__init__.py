# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains all data integrity checks."""
from .category_mismatch_train_test import CategoryMismatchTrainTest
from .conflicting_labels import ConflictingLabels
from .data_duplicates import DataDuplicates
from .dominant_frequency_change import DominantFrequencyChange
from .is_single_value import IsSingleValue
from .mixed_data_types import MixedDataTypes
from .mixed_nulls import MixedNulls
from .new_label_train_test import NewLabelTrainTest
from .outlier_sample_detection import OutlierSampleDetection
from .special_chars import SpecialCharacters
from .string_length_out_of_bounds import StringLengthOutOfBounds
from .string_mismatch import StringMismatch
from .string_mismatch_comparison import StringMismatchComparison

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
    'ConflictingLabels',
    'OutlierSampleDetection'
]
