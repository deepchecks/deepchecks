"""Module containing the train test validation check in the vision package."""
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
from .feature_label_correlation_change import FeatureLabelCorrelationChange
from .heatmap_comparison import HeatmapComparison
from .image_dataset_drift import ImageDatasetDrift
from .image_property_drift import ImagePropertyDrift
from .new_labels import NewLabels
from .similar_image_leakage import SimilarImageLeakage
from .train_test_label_drift import TrainTestLabelDrift

__all__ = [
    'HeatmapComparison',
    'ImageDatasetDrift',
    'ImagePropertyDrift',
    'NewLabels',
    'SimilarImageLeakage',
    'FeatureLabelCorrelationChange',
    'TrainTestLabelDrift'
]
