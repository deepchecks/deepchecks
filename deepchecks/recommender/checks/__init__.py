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
"""Module contains checks for recommender systems."""
from .date_train_test_leakage_overlap import DateTrainTestLeakageOverlap
from .cold_start_detection import ColdStartDetection
from .label_popularity_drift import LabelPopularityDrift
from .prediction_popularity_drift import PredictionPopularityDrift
from .sample_performance import SamplePerformance
from .user_session_length_drift import UserSessionDrift
from .segment_performance import SegmentPerformance
from .weak_segments_performance import WeakSegmentsPerformance

from .product_association import ProductAssociation
__all__ = ['LabelPopularityDrift',
           'PredictionPopularityDrift',
           'SamplePerformance',
           'DateTrainTestLeakageOverlap',
           'ColdStartDetection',
           'UserSessionDrift',
           'SegmentPerformance',
           'WeakSegmentsPerformance',
           'ProductAssociation'
           ]
