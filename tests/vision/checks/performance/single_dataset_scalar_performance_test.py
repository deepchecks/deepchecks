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

import torch
from ignite.metrics import Precision, Accuracy
from hamcrest import (assert_that, equal_to, greater_than_or_equal_to,
                      is_in, raises, calling)
import warnings
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks.performance.single_dataset_scalar_performance import SingleDatasetScalarPerformance


def test_detection_deafults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    # scalar
    assert_that(type(result.value['score']), equal_to(float))
    # metric
    assert_that(result.value['score'], greater_than_or_equal_to(0))


def test_detection_w_params(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # params that should run normally
    check = SingleDatasetScalarPerformance(reduce=torch.max, reduce_name='torch_max')
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    assert_that(type(result.value['score']), equal_to(float))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    # scalar
    assert_that(type(result.value['score']), equal_to(float))
    # metric
    assert_that(result.value['score'], greater_than_or_equal_to(0))


def test_add_condition(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance()
    check.add_condition_greater_than(0.5)
    check.add_condition_less_equal_to(0.2)

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    assert_that(result.conditions_results[0].is_pass)
    assert_that(result.conditions_results[1].is_pass, equal_to(False))


def test_classification_w_params(mnist_dataset_train, mock_trained_mnist, device):
    # params that should run normally
    check = SingleDatasetScalarPerformance(Precision(), torch.max, reduce_name='max')
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)
    assert_that(type(result.value['score']), equal_to(float))

    # params that should raise a warning but still run
    check = SingleDatasetScalarPerformance(Accuracy(), torch.min)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)
        assert issubclass(w[-1].category, SyntaxWarning)
        assert type(result.value['score']) == float

    # params that should raise an error
    check = SingleDatasetScalarPerformance(Precision())
    assert_that(calling(check.run).with_args(mnist_dataset_train, mock_trained_mnist, device=device),
                raises(DeepchecksValueError, f'The metric {Precision().__class__} return a non-scalar value, '
                                                f'please specify a reduce function or choose a different metric'))

