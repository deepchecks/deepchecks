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
from ignite.metrics import Accuracy, Precision
from sklearn.metrics import make_scorer, jaccard_score

from deepchecks.core import ConditionCategory
from deepchecks.vision.checks import SingleDatasetPerformance
from deepchecks.vision.metrics_utils import ObjectDetectionTpFpFn
from deepchecks.vision.metrics_utils.custom_scorer import CustomClassificationScorer
from tests.base.utils import equal_condition_result


def test_detection_defaults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.416, 0.001))


def test_detection_w_params(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # params that should run normally
    check = SingleDatasetPerformance(scorers={'f1': ObjectDetectionTpFpFn(evaluating_function='f1')})
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    assert_that(result.value.Value.mean(), close_to(0.505, 0.001))


def test_detection_many_scorers(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Act
    check = SingleDatasetPerformance(scorers=['f1_micro', 'fnr_per_class', 'recall_macro', 'recall_per_class'])
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device).value

    # Assert
    assert_that(result[result['Metric'] == 'f1_micro']['Value'].mean(), close_to(0.5666, 0.001))
    assert_that(result[result['Metric'] == 'f1_micro']['Value'], has_length(1))
    assert_that(result[result['Metric'] == 'fnr']['Value'].mean(), close_to(0.557, 0.001))
    assert_that(result[result['Metric'] == 'fnr']['Value'], has_length(61))
    assert_that(result[result['Metric'] == 'recall_macro']['Value'].mean(), close_to(0.443, 0.001))
    assert_that(result[result['Metric'] == 'recall_macro']['Value'], has_length(1))
    assert_that(result[result['Metric'] == 'recall']['Value'].mean(),
                close_to(result[result['Metric'] == 'recall_macro']['Value'].mean(), 0.001))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.98, 0.001))


def test_classification_custom_scorer(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    scorer = make_scorer(jaccard_score, average=None, zero_division=0)
    check = SingleDatasetPerformance(scorers={'kappa': CustomClassificationScorer(scorer)})

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device, n_samples=None)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.963, 0.001))


def test_classification_sklearn_scorers(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_per_class'])

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device, n_samples=None)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.981, 0.001))
    assert_that(result.value.Value, has_length(10))


def test_segmentation_many_scorers(segmentation_coco_train_visiondata, trained_segmentation_deeplabv3_mobilenet_model,
                                   device):
    check = SingleDatasetPerformance(scorers=['dice_per_class', 'dice_macro', 'iou_micro'])
    result = check.run(segmentation_coco_train_visiondata, trained_segmentation_deeplabv3_mobilenet_model,
                       device=device, with_display=False).value

    assert_that(result[result['Metric'] == 'dice']['Value'].mean(), close_to(0.649, 0.001))
    assert_that(result[result['Metric'] == 'dice_macro']['Value'], close_to(0.649, 0.001))
    assert_that(result[result['Metric'] == 'iou_micro']['Value'], close_to(0.948, 0.001))


def test_condition_greater_than(mnist_dataset_test, mock_trained_mnist, device):
    check = SingleDatasetPerformance().add_condition_greater_than(0.8) \
        .add_condition_greater_than(1.0, ['Precision']) \
        .add_condition_greater_than(0.5, ['Accuracy']) \
        .add_condition_greater_than(0.5, class_mode='a')

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Score is greater than 0.8 for classes: all',
        details='Passed for all of the metrics.'
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name='Score is greater than 1.0 for classes: all',
        details='Failed for metrics: [\'Precision\']'
    ))

    assert_that(result.conditions_results[2], equal_condition_result(
        is_pass=False,
        category=ConditionCategory.ERROR,
        details='Exception in condition: DeepchecksValueError: The requested metric was not calculated, the metrics '
                'calculated in this check are: [\'Precision\' \'Recall\'].',
        name='Score is greater than 0.5 for classes: all'
    ))

    assert_that(result.conditions_results[3], equal_condition_result(
        is_pass=False,
        category=ConditionCategory.ERROR,
        details='Exception in condition: DeepchecksValueError: class_mode expected be one of the classes in the check '
                'results or any or all, received a.',
        name='Score is greater than 0.5 for classes: a'
    ))


def test_reduce_output(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance(scorers={'pr': Precision(), 'ac': Accuracy()})

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device).reduce_output()

    # Assert
    assert_that(result['ac'], close_to(0.9813, 0.01))
    assert_that(result[('pr', '0')], close_to(0.978894472, 0.01))
    assert_that(result[('pr', '7')], close_to(0.972920, 0.01))
