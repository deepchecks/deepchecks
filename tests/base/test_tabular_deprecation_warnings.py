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
"""Contains unit tests for the tabular package deprecation warnings."""

import pytest

from deepchecks.tabular.checks import SegmentPerformance
from deepchecks.tabular.checks import WholeDatasetDrift


def test_deprecation_segment_performance_warning():
    with pytest.warns(DeprecationWarning, match='The SegmentPerformance check is deprecated and will be removed in the '
                                                '0.11 version. Please use the WeakSegmentsPerformance check instead.'):
        _ = SegmentPerformance()

def test_deprecation_whole_dataset_drift_warning():
    with pytest.warns(DeprecationWarning, match='The WholeDatasetDrift check is deprecated and will be removed in the '
                                                '0.11 version. Please use the MultivariateDrift check instead.'):
        _ = WholeDatasetDrift()