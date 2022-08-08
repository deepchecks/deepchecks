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
from .columns_info import ColumnsInfo
from .conflicting_labels import ConflictingLabels
from .data_duplicates import DataDuplicates
from .feature_feature_correlation import FeatureFeatureCorrelation
from .feature_label_correlation import FeatureLabelCorrelation
from .identifier_label_correlation import IdentifierLabelCorrelation
from .is_single_value import IsSingleValue
from .mixed_data_types import MixedDataTypes
from .mixed_nulls import MixedNulls
from .outlier_sample_detection import OutlierSampleDetection
from .percent_of_nulls import PercentOfNulls
from .special_chars import SpecialCharacters
from .string_length_out_of_bounds import StringLengthOutOfBounds
from .string_mismatch import StringMismatch

__all__ = [
    'ColumnsInfo',
    'MixedNulls',
    'StringMismatch',
    'MixedDataTypes',
    'IsSingleValue',
    'SpecialCharacters',
    'StringLengthOutOfBounds',
    'DataDuplicates',
    'ConflictingLabels',
    'OutlierSampleDetection',
    'FeatureLabelCorrelation',
    'FeatureFeatureCorrelation',
    'IdentifierLabelCorrelation',
    'PercentOfNulls'
]
