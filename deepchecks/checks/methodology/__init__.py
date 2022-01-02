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
"""Module contains checks for methodological flaws in the model building process."""
from .boosting_overfit import BoostingOverfit
from .unused_features import UnusedFeatures
from .single_feature_contribution import SingleFeatureContribution
from .single_feature_contribution_train_test import SingleFeatureContributionTrainTest
from .index_leakage import IndexTrainTestLeakage
from .train_test_samples_mix import TrainTestSamplesMix
from .date_train_test_leakage_duplicates import DateTrainTestLeakageDuplicates
from .date_train_test_leakage_overlap import DateTrainTestLeakageOverlap
from .identifier_leakage import IdentifierLeakage
from .model_inference_time import ModelInferenceTimeCheck
from .datasets_size_comparison import DatasetsSizeComparison


__all__ = [
    'BoostingOverfit',
    'UnusedFeatures',
    'SingleFeatureContribution',
    'SingleFeatureContributionTrainTest',
    'IndexTrainTestLeakage',
    'TrainTestSamplesMix',
    'DateTrainTestLeakageDuplicates',
    'DateTrainTestLeakageOverlap',
    'IdentifierLeakage',
    'ModelInferenceTimeCheck',
    'DatasetsSizeComparison',
]
