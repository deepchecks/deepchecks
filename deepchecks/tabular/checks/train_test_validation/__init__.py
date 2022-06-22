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
"""Module contains checks of train test validation checks."""

from deepchecks.tabular.checks.data_integrity.identifier_label_correlation import IdentifierLabelCorrelation

from .category_mismatch_train_test import CategoryMismatchTrainTest
from .datasets_size_comparison import DatasetsSizeComparison
from .date_train_test_leakage_duplicates import DateTrainTestLeakageDuplicates
from .date_train_test_leakage_overlap import DateTrainTestLeakageOverlap
from .dominant_frequency_change import DominantFrequencyChange
from .feature_label_correlation_change import FeatureLabelCorrelationChange
from .index_leakage import IndexTrainTestLeakage
from .new_label_train_test import NewLabelTrainTest
from .string_mismatch_comparison import StringMismatchComparison
from .train_test_feature_drift import TrainTestFeatureDrift
from .train_test_label_drift import TrainTestLabelDrift
from .train_test_samples_mix import TrainTestSamplesMix
from .whole_dataset_drift import WholeDatasetDrift

__all__ = [
    'CategoryMismatchTrainTest',
    'DatasetsSizeComparison',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'DominantFrequencyChange',
    'IdentifierLabelCorrelation',
    'IndexTrainTestLeakage',
    'NewLabelTrainTest',
    'FeatureLabelCorrelationChange',
    'StringMismatchComparison',
    'TrainTestFeatureDrift',
    'TrainTestLabelDrift',
    'TrainTestSamplesMix',
    'WholeDatasetDrift',
]
