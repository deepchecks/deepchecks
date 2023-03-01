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
from hamcrest import assert_that, calling, close_to, greater_than, has_length, raises

from deepchecks.core.errors import DeepchecksNotSupportedError, ModelValidationError
from deepchecks.vision.checks import MeanAveragePrecisionReport
from tests.base.utils import equal_condition_result


def test_mnist_error(mnist_visiondata_test):
    # Arrange
    check = MeanAveragePrecisionReport()
    # Act
    assert_that(
        calling(check.run).with_args(mnist_visiondata_test),
        raises(DeepchecksNotSupportedError, r'Check is irrelevant for task of type TaskType.CLASSIFICATION')
    )


def test_coco(coco_visiondata_test):
    # Arrange
    check = MeanAveragePrecisionReport() \
            .add_condition_mean_average_precision_greater_than(0.1) \
            .add_condition_mean_average_precision_greater_than(0.4) \
            .add_condition_average_mean_average_precision_greater_than() \
            .add_condition_average_mean_average_precision_greater_than(0.5)

    # Act
    result = check.run(coco_visiondata_test)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'mAP@[.50::.95] (avg.%)'], close_to(0.409, 0.001))
    assert_that(df.loc['All', 'mAP@.50 (%)'], close_to(0.566, 0.001))
    assert_that(df.loc['All', 'mAP@.75 (%)'], close_to(0.425, 0.001))

    assert_that(df.loc['Small (area < 32^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.212, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'mAP@.50 (%)'], close_to(0.342, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'mAP@.75 (%)'], close_to(0.212, 0.001))

    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.383, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'mAP@.50 (%)'], close_to(0.600, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'mAP@.75 (%)'], close_to(0.349, 0.001))

    assert_that(df.loc['Large (area < 96^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.541, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'mAP@.50 (%)'], close_to(0.674, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'mAP@.75 (%)'], close_to(0.585, 0.001))

    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        details='Found lowest score of 0.21 for area Small (area < 32^2) and IoU mAP@[.50::.95] (avg.%)',
        name='Scores are greater than 0.1'
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name='Scores are greater than 0.4',
        details='Found lowest score of 0.21 for area Small (area < 32^2) and IoU mAP@[.50::.95] (avg.%)'
    ))

    assert_that(result.conditions_results[2], equal_condition_result(
        is_pass=True,
        details='mAP score is: 0.41',
        name='mAP score is greater than 0.3'
    ))

    assert_that(result.conditions_results[3], equal_condition_result(
        is_pass=False,
        name='mAP score is greater than 0.5',
        details="mAP score is: 0.41"
    ))


def test_coco_area_param(coco_visiondata_test):
    # Arrange
    check = MeanAveragePrecisionReport(area_range=(40**2, 100**2))

    # Act
    result = check.run(coco_visiondata_test)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'mAP@[.50::.95] (avg.%)'], close_to(0.409, 0.001))
    assert_that(df.loc['All', 'mAP@.50 (%)'], close_to(0.566, 0.001))
    assert_that(df.loc['All', 'mAP@.75 (%)'], close_to(0.425, 0.001))

    assert_that(df.loc['Small (area < 40^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.191, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'mAP@.50 (%)'], close_to(0.324, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'mAP@.75 (%)'], close_to(0.179, 0.001))

    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.414, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'mAP@.50 (%)'], close_to(0.622, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'mAP@.75 (%)'], close_to(0.388, 0.001))

    assert_that(df.loc['Large (area < 100^2)', 'mAP@[.50::.95] (avg.%)'], close_to(0.542, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'mAP@.50 (%)'], close_to(0.673, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'mAP@.75 (%)'], close_to(0.592, 0.001))
    assert_that(result.display, has_length(greater_than(0)))


def test_coco_area_param_without_display(coco_visiondata_test):
    # Arrange
    check = MeanAveragePrecisionReport(area_range=(40**2, 100**2))

    # Act
    result = check.run(coco_visiondata_test, with_display=False)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'mAP@[.50::.95] (avg.%)'], close_to(0.409, 0.001))
    assert_that(result.display, has_length(0))
