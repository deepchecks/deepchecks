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

from hamcrest import assert_that, close_to, has_length
from deepchecks.vision.checks import WeakSegmentsPerformance


def test_detection_defaults(coco_train_visiondata, device):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(coco_train_visiondata, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.416, 0.001))