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

import torch
from hamcrest import assert_that, calling, close_to, equal_to, greater_than_or_equal_to, raises, has_items
from ignite.metrics import Accuracy, Precision

from base.utils import equal_condition_result
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import SingleDatasetScalarPerformance


def test_detection_defaults(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)

    # Assert
    # scalar
    assert_that(type(result.value['score']), equal_to(float))
    # metric
    assert_that(result.value['score'], close_to(0.39, 0.001))


def test_detection_w_params(coco_train_visiondata, mock_trained_yolov5_object_detection, device):
    # params that should run normally
    check = SingleDatasetScalarPerformance(reduce=torch.nansum, reduce_name='nan_sum')
    result = check.run(coco_train_visiondata, mock_trained_yolov5_object_detection, device=device)
    assert_that(type(result.value['score']), equal_to(float))
    assert_that(result.value['score'], close_to(24.596, 0.001))


def test_classification_defaults(mnist_dataset_train, mock_trained_mnist, device):
    # Arrange
    check = SingleDatasetScalarPerformance()

    # Act
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)

    # Assert
    # scalar
    assert_that(type(result.value['score']), equal_to(float))
    # metric
    assert_that(result.value['score'], close_to(0.98, 0.001))


def test_classification_w_params(mnist_dataset_train, mock_trained_mnist, device):
    # params that should run normally
    check = SingleDatasetScalarPerformance(Precision(), torch.max, reduce_name='max')
    check.add_condition_greater_than(0.5)
    check.add_condition_less_or_equal(0.2)
    result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)
    assert_that(type(result.value['score']), equal_to(float))
    assert_that(result.value['score'], close_to(0.993, 0.001))
    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='The score Precision is 0.99',
                               name='Score is greater than 0.5'),
        equal_condition_result(is_pass=False,
                               details='The score Precision is 0.99',
                               name='Score is less or equal to 0.2'))
    )

    # params that should raise a warning but still run
    check = SingleDatasetScalarPerformance(Accuracy(), torch.min)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        result = check.run(mnist_dataset_train, mock_trained_mnist, device=device)
        assert issubclass(w[-1].category, SyntaxWarning)
        assert isinstance(result.value['score'], float)

    # params that should raise an error
    check = SingleDatasetScalarPerformance(Precision())
    assert_that(calling(check.run).with_args(mnist_dataset_train, mock_trained_mnist, device=device),
                raises(DeepchecksValueError, f'The metric {Precision().__class__} return a non-scalar value, '
                                             f'please specify a reduce function or choose a different metric'))

