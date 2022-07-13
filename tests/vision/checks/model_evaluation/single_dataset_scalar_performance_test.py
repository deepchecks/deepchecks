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
from deepchecks.vision.metrics_utils import ObjectDetectionTpFpFn

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import SingleDatasetScalarPerformance
from deepchecks.vision.metrics_utils.custom_scorer import CustomScorer
from tests.base.utils import equal_condition_result


def test_detection_defaults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.416, 0.001))


def test_detection_w_params(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # params that should run normally
    check = SingleDatasetScalarPerformance(alternative_scorers={'f1': ObjectDetectionTpFpFn(evaluting_function='f1')})
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    assert_that(result.value.Value.mean(), close_to(0.505, 0.001))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.98, 0.001))


def test_classification_custom_scorer(mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance(alternative_scorers={'kappa': CustomScorer(cohen_kappa_score)})

    # Act
    result = check.run(mnist_dataset_test, mock_trained_mnist, device=device, n_samples=None)

    # Assert
    assert_that(result.value.Value.mean(), close_to(0.979, 0.001))
