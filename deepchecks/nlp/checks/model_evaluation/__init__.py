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
"""Module containing the model evaluation checks in the nlp package."""

from deepchecks.nlp.checks.model_evaluation.single_dataset_performance import SingleDatasetPerformance
from deepchecks.nlp.checks.model_evaluation.train_test_prediction_drift import TrainTestPredictionDrift
from deepchecks.nlp.checks.model_evaluation.weak_segments_performance import AdditionalDataSegmentsPerformance, \
    PropertySegmentsPerformance

__all__ = [
    'SingleDatasetPerformance', 'AdditionalDataSegmentsPerformance', 'PropertySegmentsPerformance',
    'TrainTestPredictionDrift'
]
