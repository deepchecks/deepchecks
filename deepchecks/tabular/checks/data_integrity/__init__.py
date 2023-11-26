# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains all data integrity checks."""
from deepchecks.tabular.checks.data_integrity.class_imbalance import ClassImbalance
from deepchecks.tabular.checks.data_integrity.columns_info import ColumnsInfo
from deepchecks.tabular.checks.data_integrity.conflicting_labels import ConflictingLabels
from deepchecks.tabular.checks.data_integrity.data_duplicates import DataDuplicates
from deepchecks.tabular.checks.data_integrity.feature_feature_correlation import FeatureFeatureCorrelation
from deepchecks.tabular.checks.data_integrity.feature_label_correlation import FeatureLabelCorrelation
from deepchecks.tabular.checks.data_integrity.identifier_label_correlation import IdentifierLabelCorrelation
from deepchecks.tabular.checks.data_integrity.is_single_value import IsSingleValue
from deepchecks.tabular.checks.data_integrity.mixed_data_types import MixedDataTypes
from deepchecks.tabular.checks.data_integrity.mixed_nulls import MixedNulls
from deepchecks.tabular.checks.data_integrity.outlier_sample_detection import OutlierSampleDetection
from deepchecks.tabular.checks.data_integrity.percent_of_nulls import PercentOfNulls
from deepchecks.tabular.checks.data_integrity.special_chars import SpecialCharacters
from deepchecks.tabular.checks.data_integrity.string_length_out_of_bounds import StringLengthOutOfBounds
from deepchecks.tabular.checks.data_integrity.string_mismatch import StringMismatch

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
    'ClassImbalance',
    'OutlierSampleDetection',
    'FeatureLabelCorrelation',
    'FeatureFeatureCorrelation',
    'IdentifierLabelCorrelation',
    'PercentOfNulls'
]
