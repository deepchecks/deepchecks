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
"""Module contains checks of train test validation checks."""

from deepchecks.tabular.checks.data_integrity.identifier_label_correlation import IdentifierLabelCorrelation
from deepchecks.tabular.checks.train_test_validation.category_mismatch_train_test import CategoryMismatchTrainTest
from deepchecks.tabular.checks.train_test_validation.datasets_size_comparison import DatasetsSizeComparison
from deepchecks.tabular.checks.train_test_validation.date_train_test_leakage_duplicates import \
    DateTrainTestLeakageDuplicates
from deepchecks.tabular.checks.train_test_validation.date_train_test_leakage_overlap import DateTrainTestLeakageOverlap
from deepchecks.tabular.checks.train_test_validation.feature_drift import FeatureDrift
from deepchecks.tabular.checks.train_test_validation.feature_label_correlation_change import \
    FeatureLabelCorrelationChange
from deepchecks.tabular.checks.train_test_validation.index_leakage import IndexTrainTestLeakage
from deepchecks.tabular.checks.train_test_validation.label_drift import LabelDrift
from deepchecks.tabular.checks.train_test_validation.multivariate_drift import MultivariateDrift
from deepchecks.tabular.checks.train_test_validation.new_category_train_test import NewCategoryTrainTest
from deepchecks.tabular.checks.train_test_validation.new_label_train_test import NewLabelTrainTest
from deepchecks.tabular.checks.train_test_validation.string_mismatch_comparison import StringMismatchComparison
from deepchecks.tabular.checks.train_test_validation.train_test_feature_drift import TrainTestFeatureDrift
from deepchecks.tabular.checks.train_test_validation.train_test_label_drift import TrainTestLabelDrift
from deepchecks.tabular.checks.train_test_validation.train_test_samples_mix import TrainTestSamplesMix
from deepchecks.tabular.checks.train_test_validation.whole_dataset_drift import WholeDatasetDrift

__all__ = [
    'CategoryMismatchTrainTest',
    'NewCategoryTrainTest',
    'DatasetsSizeComparison',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLabelCorrelation',
    'IndexTrainTestLeakage',
    'NewLabelTrainTest',
    'FeatureLabelCorrelationChange',
    'StringMismatchComparison',
    'TrainTestFeatureDrift',
    'FeatureDrift',
    'TrainTestLabelDrift',
    'LabelDrift',
    'TrainTestSamplesMix',
    'MultivariateDrift',
    'WholeDatasetDrift'
]
