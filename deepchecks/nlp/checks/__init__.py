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

from deepchecks.nlp.checks.data_integrity import (ConflictingLabels, FrequentSubstrings, PropertyLabelCorrelation,
                                                  SpecialCharacters, TextDuplicates, TextPropertyOutliers,
                                                  UnderAnnotatedMetaDataSegments, UnderAnnotatedPropertySegments,
                                                  UnknownTokens)
from deepchecks.nlp.checks.model_evaluation import (ConfusionMatrixReport, MetadataSegmentsPerformance, PredictionDrift,
                                                    PropertySegmentsPerformance, SingleDatasetPerformance,
                                                    TrainTestPerformance)
from deepchecks.nlp.checks.train_test_validation import (LabelDrift, PropertyDrift, TextEmbeddingsDrift,
                                                         TrainTestSamplesMix)

__all__ = [
    # Data Integrity
    'PropertyLabelCorrelation',
    'TextPropertyOutliers',
    'TextDuplicates',
    'ConflictingLabels',
    'SpecialCharacters',
    'UnknownTokens',
    'UnderAnnotatedMetaDataSegments',
    'UnderAnnotatedPropertySegments',
    'FrequentSubstrings',

    # Model Evaluation
    'SingleDatasetPerformance',
    'MetadataSegmentsPerformance',
    'PropertySegmentsPerformance',
    'ConfusionMatrixReport',
    'TrainTestPerformance',
    'PredictionDrift',

    # Train Test Validation
    'LabelDrift',
    'PropertyDrift',
    'TextEmbeddingsDrift',
    'TrainTestSamplesMix'
]
