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
from deepchecks.vision.checks.performance.class_performance import ClassPerformance

import torch.nn as nn

from hamcrest import assert_that, close_to, equal_to, is_in


def test_mnist_largest(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       device=device)
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.991532, 0.05))
    assert_that(first_row['Number of samples'], equal_to(6742))
    assert_that(first_row['Class'], equal_to(1))


def test_mnist_smallest(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='smallest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       device=device)
    first_row = result.value.sort_values(by='Number of samples', ascending=True).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.988739, 0.05))
    assert_that(first_row['Number of samples'], equal_to(892))
    assert_that(first_row['Class'], equal_to(5))


def test_mnist_worst(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='worst')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       device=device)
    first_row = result.value.loc[result.value['Metric'] == 'Precision'].sort_values(by='Value', ascending=True).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.977713, 0.05))


def test_mnist_best(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='best')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       device=device)
    first_row = result.value.loc[result.value['Metric'] == 'Precision'].sort_values(by='Value', ascending=False).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.990854, 0.05))


def test_coco_best(coco_train_visiondata, coco_test_visiondata, trained_yolov5_object_detection, device):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='best')
    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata,
                       trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(len(result.value), equal_to(8))

    first_row = result.value.loc[result.value['Metric'] == 'AP'].sort_values(by='Value', ascending=False).iloc[0]
    assert_that(first_row['Value'], close_to(0.990854, 0.05))
    assert_that(first_row['Number of samples'], equal_to(3))
    assert_that(first_row['Class'], is_in([29]))

    first_row = result.value.loc[result.value['Metric'] == 'AR'].sort_values(by='Value', ascending=False).iloc[0]
    assert_that(first_row['Value'], close_to(0.990854, 0.05))
    assert_that(first_row['Number of samples'], equal_to(3))
    assert_that(first_row['Class'], is_in([29]))


def test_class_list(mnist_dataset_train, mnist_dataset_test, trained_mnist, device):
    # Arrange

    check = ClassPerformance(class_list_to_show=[1])
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, trained_mnist,
                       device=device)

    # Assert
    assert_that(len(result.value), equal_to(4))
    assert_that(result.value['Class'].iloc[0], equal_to(1))
    assert_that(result.value['Class'].iloc[1], equal_to(1))
    assert_that(result.value['Class'].iloc[2], equal_to(1))
    assert_that(result.value['Class'].iloc[3], equal_to(1))
