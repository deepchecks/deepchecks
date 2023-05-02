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
"""Module importing all nlp checks."""

from deepchecks.nlp.checks.data_integrity import (ConflictingLabels, PropertyLabelCorrelation, SpecialCharacters,
                                                  TextDuplicates, TextPropertyOutliers, UnknownTokens)
from deepchecks.nlp.checks.model_evaluation import (ConfusionMatrixReport, MetadataSegmentsPerformance, PredictionDrift,
                                                    PropertySegmentsPerformance, SingleDatasetPerformance,
                                                    TrainTestPerformance)
from deepchecks.nlp.checks.train_test_validation import LabelDrift, PropertyDrift, TrainTestSamplesMix

__all__ = [
    # Data Integrity
    'PropertyLabelCorrelation',
    'TextPropertyOutliers',
    'TextDuplicates',
    'ConflictingLabels',
    'SpecialCharacters',
    'UnknownTokens',

    # Model Evaluation
    'SingleDatasetPerformance',
    'MetadataSegmentsPerformance',
    'PropertySegmentsPerformance',
    'ConfusionMatrixReport',
    'TrainTestPerformance',

    # Train Test Validation
    'PredictionDrift',
    'LabelDrift',
    'PropertyDrift',
    'TrainTestSamplesMix'
]
