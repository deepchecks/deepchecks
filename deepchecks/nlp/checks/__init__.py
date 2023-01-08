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
"""Module importing all nlp checks."""

from deepchecks.nlp.checks.data_integrity import TextPropertyOutliers
from deepchecks.nlp.checks.model_evaluation import SingleDatasetPerformance, TrainTestPredictionDrift
from deepchecks.nlp.checks.train_test_validation import KeywordFrequencyDrift, TrainTestLabelDrift, TextEmbeddingsDrift

__all__ = [
    # Data Integrity
    'TextPropertyOutliers',

    # Train-Test Validation
    'KeywordFrequencyDrift',
    'TrainTestLabelDrift',
    'TextEmbeddingsDrift',

    # Model Evaluation
    'SingleDatasetPerformance',
    'TrainTestPredictionDrift',
]
