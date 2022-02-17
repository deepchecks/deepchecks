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
import re
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

    assert_that(df.loc['All', 'mAP@0.5..0.95 (%)'], close_to(0.361, 0.001))
    assert_that(df.loc['All', 'AP@.50 (%)'], close_to(0.501, 0.001))
    assert_that(df.loc['All', 'AP@.75 (%)'], close_to(0.376, 0.001))

    assert_that(df.loc['Small (area<32^2)', 'mAP@0.5..0.95 (%)'], close_to(0.188, 0.001))
    assert_that(df.loc['Small (area<32^2)', 'AP@.50 (%)'], close_to(0.288, 0.001))
    assert_that(df.loc['Small (area<32^2)', 'AP@.75 (%)'], close_to(0.193, 0.001))

    assert_that(df.loc['Medium (32^2<area<96^2)', 'mAP@0.5..0.95 (%)'], close_to(0.367, 0.001))
    assert_that(df.loc['Medium (32^2<area<96^2)', 'AP@.50 (%)'], close_to(0.568, 0.001))
    assert_that(df.loc['Medium (32^2<area<96^2)', 'AP@.75 (%)'], close_to(0.369, 0.001))

    assert_that(df.loc['Large (area<96^2)', 'mAP@0.5..0.95 (%)'], close_to(0.476, 0.001))
    assert_that(df.loc['Large (area<96^2)', 'AP@.50 (%)'], close_to(0.575, 0.001))
    assert_that(df.loc['Large (area<96^2)', 'AP@.75 (%)'], close_to(0.533, 0.001))

    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Scores are not less than 0.1'
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name='Scores are not less than 0.4',
        details="Found scores below threshold:\n{'All': {'mAP@0.5..0.95 (%)': '0.361'}, " + \
                "'Small (area<32^2)': {'AP@.50 (%)': '0.288'}, 'Medium (32^2<area<96^2)': {'AP@.75 (%)': '0.369'}}"
    ))
