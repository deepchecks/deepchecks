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

import warnings

import pandas as pd
import torch
from hamcrest import assert_that, calling, close_to, equal_to, greater_than_or_equal_to, has_items, none, raises
from ignite.metrics import Accuracy, Precision
from sklearn.metrics import cohen_kappa_score

from deepchecks.core import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import SingleDatasetPerformance
from deepchecks.vision.metrics_utils import ObjectDetectionTpFpFn
from deepchecks.vision.metrics_utils.custom_scorer import CustomScorer
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
    check = SingleDatasetPerformance(scorers={'f1': ObjectDetectionTpFpFn(evaluting_function='f1')})
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    assert_that(result.value.Value.mean(), close_to(0.505, 0.001))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.98, 0.001))


def test_classification_custom_scorer(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance(scorers={'kappa': CustomScorer(cohen_kappa_score)})

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device, n_samples=None)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.979, 0.001))


def test_condition_greater_than(mnist_dataset_test, mock_trained_mnist, device):
    check = SingleDatasetPerformance().add_condition_greater_than(0.8)\
        .add_condition_greater_than(1.0, ['Precision'])\
        .add_condition_greater_than(0.5, ['Accuracy'])\
        .add_condition_greater_than(0.5, class_mode='a')

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name='Score is greater than 0.8 for classes: all',
        details='Passed for all of the mertics.'
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
                'results or any or all, recieved a.',
        name='Score is greater than 0.5 for classes: a'
        ))


def test_reduce_output(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetPerformance(scorers={'pr': Precision(), 'ac': Accuracy()})

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device).reduce_output()

    # Assert
    assert_that(result, equal_to({
         'ac': 0.9813,
         'pr_0': 0.978894472361809,
         'pr_1': 0.9807524059492564,
         'pr_2': 0.9845559845559846,
         'pr_3': 0.9773844641101278,
         'pr_4': 0.9886714727085479,
         'pr_5': 0.975363941769317,
         'pr_6': 0.9895068205666316,
         'pr_7': 0.9729206963249516,
         'pr_8': 0.9905660377358491,
         'pr_9': 0.9750996015936255
    }))
