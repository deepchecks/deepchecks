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
"""Test functions of the heatmap comparison check."""
from hamcrest import assert_that, has_entries, close_to, equal_to, raises, calling

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks.distribution import HeatmapComparison


def test_with_drift_object_detection_alternative_measurements(coco_train_visiondata, coco_test_visiondata):
    # Arrange
    check = HeatmapComparison()
    check.run(coco_train_visiondata, coco_test_visiondata)
