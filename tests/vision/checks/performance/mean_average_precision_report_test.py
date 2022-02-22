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
from hamcrest import assert_that, close_to, has_length, calling, raises

from tests.checks.utils import equal_condition_result
from deepchecks.core.errors import ModelValidationError
from deepchecks.vision.checks.performance import MeanAveragePrecisionReport
from deepchecks.vision.datasets.detection.coco import yolo_prediction_formatter
from deepchecks.vision.utils import DetectionPredictionFormatter


def test_mnist_error(mnist_dataset_test, trained_mnist):
    # Arrange
    check = MeanAveragePrecisionReport()
    # Act
    assert_that(
        calling(check.run).with_args(mnist_dataset_test, trained_mnist),
            raises(ModelValidationError, r'Check is irrelevant for task of type TaskType.CLASSIFICATION')
    )


def test_coco(coco_test_visiondata, trained_yolov5_object_detection):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = MeanAveragePrecisionReport() \
            .add_condition_test_average_precision_not_less_than(0.1) \
            .add_condition_test_average_precision_not_less_than(0.4)

    # Act
    result = check.run(coco_test_visiondata,
                       trained_yolov5_object_detection, prediction_formatter=pred_formatter)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'mAP@0.5..0.95 (%)'], close_to(0.409, 0.001))
    assert_that(df.loc['All', 'AP@.50 (%)'], close_to(0.566, 0.001))
    assert_that(df.loc['All', 'AP@.75 (%)'], close_to(0.425, 0.001))

    assert_that(df.loc['Small (area < 32^2)', 'mAP@0.5..0.95 (%)'], close_to(0.212, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AP@.50 (%)'], close_to(0.342, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AP@.75 (%)'], close_to(0.212, 0.001))

    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'mAP@0.5..0.95 (%)'], close_to(0.383, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AP@.50 (%)'], close_to(0.600, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AP@.75 (%)'], close_to(0.349, 0.001))

    assert_that(df.loc['Large (area < 96^2)', 'mAP@0.5..0.95 (%)'], close_to(0.541, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AP@.50 (%)'], close_to(0.674, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AP@.75 (%)'], close_to(0.585, 0.001))

    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Scores are not less than 0.1'
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name='Scores are not less than 0.4',
        details="Found scores below threshold:\n{'Small (area < 32^2)': {'AP@.50 (%)': '0.342'}, "
                "'Medium (32^2 < area < 96^2)': {'AP@.75 (%)': '0.35'}}"
    ))


def test_coco_area_param(coco_test_visiondata, trained_yolov5_object_detection):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = MeanAveragePrecisionReport(area_range=(40**2, 100**2))

    # Act
    result = check.run(coco_test_visiondata,
                       trained_yolov5_object_detection, prediction_formatter=pred_formatter)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'mAP@0.5..0.95 (%)'], close_to(0.409, 0.001))
    assert_that(df.loc['All', 'AP@.50 (%)'], close_to(0.566, 0.001))
    assert_that(df.loc['All', 'AP@.75 (%)'], close_to(0.425, 0.001))

    assert_that(df.loc['Small (area < 40^2)', 'mAP@0.5..0.95 (%)'], close_to(0.191, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AP@.50 (%)'], close_to(0.324, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AP@.75 (%)'], close_to(0.179, 0.001))

    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'mAP@0.5..0.95 (%)'], close_to(0.414, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AP@.50 (%)'], close_to(0.622, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AP@.75 (%)'], close_to(0.388, 0.001))

    assert_that(df.loc['Large (area < 100^2)', 'mAP@0.5..0.95 (%)'], close_to(0.542, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AP@.50 (%)'], close_to(0.673, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AP@.75 (%)'], close_to(0.592, 0.001))
