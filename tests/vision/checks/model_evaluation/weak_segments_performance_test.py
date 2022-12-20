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
from hamcrest import assert_that, calling, close_to, has_items, raises

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision.checks import WeakSegmentsPerformance
from tests.base.utils import equal_condition_result


def test_detection_defaults(coco_visiondata_train):
    # Arrange
    check = WeakSegmentsPerformance()

    # Act
    result = check.run(coco_visiondata_train)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.691, 0.001))


def test_detection_condition(coco_visiondata_train):
    check = WeakSegmentsPerformance().add_condition_segments_relative_performance_greater_than(0.5)

    result = check.run(coco_visiondata_train)
    condition_result = result.conditions_results

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(
            is_pass=True,
            name='The relative performance of weakest segment is greater than 50% of average model performance.',
            details='Found a segment with mean IoU score of 0.511 in comparison to an average score of 0.691 in '
                    'sampled data.')
    ))


def test_classification_defaults(mnist_visiondata_train):
    # Arrange
    check = WeakSegmentsPerformance(n_samples=1000)

    # Act
    result = check.run(mnist_visiondata_train)

    # Assert
    assert_that(result.value['avg_score'], close_to(0.082, 0.001))


def test_segmentation_defaults(segmentation_coco_visiondata_test):
    check = WeakSegmentsPerformance()
    assert_that(calling(check.run).with_args(segmentation_coco_visiondata_test),
                raises(DeepchecksProcessError))
