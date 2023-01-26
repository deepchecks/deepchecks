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

from hamcrest import assert_that, close_to, equal_to, has_length
from ignite.metrics import Accuracy, Precision
from sklearn.metrics import jaccard_score, make_scorer

from deepchecks.core import ConditionCategory
from deepchecks.vision.checks import SingleDatasetPerformance
from deepchecks.vision.metrics_utils import ObjectDetectionTpFpFn
from deepchecks.vision.metrics_utils.custom_scorer import CustomClassificationScorer
from tests.base.utils import equal_condition_result


def test_detection_defaults(coco_visiondata_train):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(coco_visiondata_train)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.393, 0.01))


def test_detection_w_params(coco_visiondata_train):
    # params that should run normally
    check = SingleDatasetPerformance(scorers={'f1': ObjectDetectionTpFpFn(evaluating_function='f1')})
    result = check.run(coco_visiondata_train)
    assert_that(result.value.Value.mean(), close_to(0.505, 0.01))


def test_detection_many_scorers(coco_visiondata_train):
    # Act
    check = SingleDatasetPerformance(scorers=['f1_micro', 'fnr_per_class', 'recall_macro', 'recall_per_class'])
    result = check.run(coco_visiondata_train).value

    # Assert
    assert_that(result[result['Metric'] == 'f1_micro']['Value'].mean(), close_to(0.571, 0.01))
    assert_that(result[result['Metric'] == 'f1_micro']['Value'], has_length(1))
    assert_that(result[result['Metric'] == 'fnr']['Value'].mean(), close_to(0.554, 0.01))
    assert_that(result[result['Metric'] == 'fnr']['Value'], has_length(61))
    assert_that(result[result['Metric'] == 'recall_macro']['Value'].mean(), close_to(0.446, 0.01))
    assert_that(result[result['Metric'] == 'recall_macro']['Value'], has_length(1))
    assert_that(result[result['Metric'] == 'recall']['Value'].mean(),
                close_to(result[result['Metric'] == 'recall_macro']['Value'].mean(), 0.01))


def test_classification_defaults(mnist_visiondata_train):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(mnist_visiondata_train)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.975, 0.01))


def test_classification_custom_scorer(mnist_visiondata_test):
    # Arrange
    scorer = make_scorer(jaccard_score, average=None, zero_division=0)
    check = SingleDatasetPerformance(scorers={'kappa': scorer})

    # Act
    result = check.run(mnist_visiondata_test)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.975, 0.01))


def test_classification_sklearn_scorers(mnist_visiondata_test):
    # Arrange
    check = SingleDatasetPerformance(scorers=['f1_per_class'])

    # Act
    result = check.run(mnist_visiondata_test)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.99, 0.01))
    assert_that(result.value.Value, has_length(10))


def test_segmentation_many_scorers(segmentation_coco_visiondata_train):
    check = SingleDatasetPerformance(scorers=['dice_per_class', 'dice_macro', 'iou_micro'])
    result = check.run(segmentation_coco_visiondata_train, with_display=False).value

    assert_that(result[result['Metric'] == 'dice']['Value'].mean(), close_to(0.649, 0.006))
    assert_that(result[result['Metric'] == 'dice_macro']['Value'], close_to(0.649, 0.006))
    assert_that(result[result['Metric'] == 'iou_micro']['Value'], close_to(0.948, 0.001))


def test_condition_greater_than(mnist_visiondata_test):
    check = SingleDatasetPerformance().add_condition_greater_than(0.8) \
        .add_condition_greater_than(1.0, ['Precision']) \
        .add_condition_greater_than(0.5, ['Accuracy']) \
        .add_condition_greater_than(0.5, class_mode='a')

    # Act
    result = check.run(mnist_visiondata_test)

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


def test_reduce_output(mnist_visiondata_test):
    # Arrange
    check = SingleDatasetPerformance(scorers={'pr': CustomClassificationScorer('precision_per_class'),
                                              'ac': CustomClassificationScorer('accuracy')})

    # Act
    result = check.run(mnist_visiondata_test).reduce_output()

    # Assert
    assert_that(result['ac'], close_to(0.99, 0.01))
    assert_that(result[('pr', '0')], close_to(1, 0.01))
    assert_that(result[('pr', '9')], close_to(0.954, 0.01))

    assert_that(check.greater_is_better(), equal_to(True))
