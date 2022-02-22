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
from deepchecks.vision.checks.performance import MeanAverageRecallReport
from deepchecks.vision.datasets.detection.coco import yolo_prediction_formatter
from deepchecks.vision.utils import DetectionPredictionFormatter


def test_mnist_error(mnist_dataset_test, trained_mnist):
    # Arrange
    check = MeanAverageRecallReport()
    # Act
    assert_that(
        calling(check.run).with_args(mnist_dataset_test, trained_mnist),
            raises(ModelValidationError, r'Check is irrelevant for task of type TaskType.CLASSIFICATION')
    )


def test_coco(coco_test_visiondata, trained_yolov5_object_detection):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = MeanAverageRecallReport() \
            .add_condition_test_average_recall_not_less_than(0.1) \
            .add_condition_test_average_recall_not_less_than(0.4)

    # Act
    result = check.run(coco_test_visiondata,
                       trained_yolov5_object_detection, prediction_formatter=pred_formatter)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'AR@1 (%)'], close_to(0.330, 0.001))
    assert_that(df.loc['All', 'AR@10 (%)'], close_to(0.423, 0.001))
    assert_that(df.loc['All', 'AR@100 (%)'], close_to(0.429, 0.001))

    assert_that(df.loc['Small (area < 32^2)', 'AR@1 (%)'], close_to(0.104, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AR@10 (%)'], close_to(0.220, 0.001))
    assert_that(df.loc['Small (area < 32^2)', 'AR@100 (%)'], close_to(0.220, 0.001))

    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@1 (%)'], close_to(0.325, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@10 (%)'], close_to(0.417, 0.001))
    assert_that(df.loc['Medium (32^2 < area < 96^2)', 'AR@100 (%)'], close_to(0.423, 0.001))

    assert_that(df.loc['Large (area < 96^2)', 'AR@1 (%)'], close_to(0.481, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AR@10 (%)'], close_to(0.544, 0.001))
    assert_that(df.loc['Large (area < 96^2)', 'AR@100 (%)'], close_to(0.549, 0.001))

    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Scores are not less than 0.1'
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name='Scores are not less than 0.4',
        details="Found scores below threshold:\n{'All': {'AR@1 (%)': '0.331'}, "
                "'Small (area < 32^2)': {'AR@10 (%)': '0.221'}}"
    ))


def test_coco_area_param(coco_test_visiondata, trained_yolov5_object_detection):
    # Arrange
    pred_formatter = DetectionPredictionFormatter(yolo_prediction_formatter)
    check = MeanAverageRecallReport(area_range=(40**2, 100**2))

    # Act
    result = check.run(coco_test_visiondata,
                       trained_yolov5_object_detection, prediction_formatter=pred_formatter)

    # Assert
    df = result.value
    assert_that(df, has_length(4))

    assert_that(df.loc['All', 'AR@1 (%)'], close_to(0.330, 0.001))
    assert_that(df.loc['All', 'AR@10 (%)'], close_to(0.423, 0.001))
    assert_that(df.loc['All', 'AR@100 (%)'], close_to(0.429, 0.001))

    assert_that(df.loc['Small (area < 40^2)', 'AR@1 (%)'], close_to(0.101, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AR@10 (%)'], close_to(0.197, 0.001))
    assert_that(df.loc['Small (area < 40^2)', 'AR@100 (%)'], close_to(0.204, 0.001))

    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@1 (%)'], close_to(0.314, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@10 (%)'], close_to(0.428, 0.001))
    assert_that(df.loc['Medium (40^2 < area < 100^2)', 'AR@100 (%)'], close_to(0.446, 0.001))

    assert_that(df.loc['Large (area < 100^2)', 'AR@1 (%)'], close_to(0.482, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AR@10 (%)'], close_to(0.547, 0.001))
    assert_that(df.loc['Large (area < 100^2)', 'AR@100 (%)'], close_to(0.551, 0.001))
