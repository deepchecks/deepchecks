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
from hamcrest import assert_that, calling, close_to, equal_to, is_, is_in, raises, has_items
from ignite.metrics import Precision, Recall

from base.utils import equal_condition_result
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.vision.checks import ClassPerformance


def test_mnist_average_error_error(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = ClassPerformance(alternative_metrics={'p': Precision(average=True)})
    # Act
    assert_that(
        calling(check.run
                ).with_args(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                            device=device),
        raises(DeepchecksValueError,
               r'The metric p returned a <class \'float\'> instead of an array/tensor')
    )


def test_mnist_largest(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='largest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device, n_samples=None)
    first_row = result.value.sort_values(by='Number of samples', ascending=False).iloc[0]
    # Assert
    assert_that(len(set(result.value['Class'])), equal_to(2))
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.977, 0.001))
    assert_that(first_row['Number of samples'], equal_to(6742))
    assert_that(first_row['Class'], equal_to(1))


def test_mnist_smallest(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='smallest')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)
    first_row = result.value.sort_values(by='Number of samples', ascending=True).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.988739, 0.05))
    assert_that(first_row['Number of samples'], equal_to(892))
    assert_that(first_row['Class'], equal_to(5))


def test_mnist_worst(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='worst')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)
    first_row = result.value.loc[result.value['Metric'] == 'Precision'].sort_values(by='Value', ascending=True).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.977713, 0.05))


def test_mnist_best(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2, show_only='best')
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device, n_samples=None)
    first_row = result.value.loc[result.value['Metric'] == 'Precision'].sort_values(by='Value', ascending=False).iloc[0]

    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(first_row['Value'], close_to(0.990, 0.001))


def test_mnist_alt(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange

    check = ClassPerformance(n_to_show=2,
                             alternative_metrics={'p': Precision(), 'r': Recall()})
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)
    p_row = result.value.loc[result.value['Metric'] == 'p'].sort_values(by='Value', ascending=False).iloc[0]
    r_row = result.value.loc[result.value['Metric'] == 'r'].sort_values(by='Value', ascending=False).iloc[0]
    # Assert
    assert_that(len(result.value), equal_to(8))
    assert_that(p_row['Value'], close_to(.984, 0.001))
    assert_that(r_row['Value'], close_to(0.988, 0.001))


def test_coco_best(coco_train_visiondata, coco_test_visiondata, mock_trained_yolov5_object_detection, device):
    # Arrange
    check = ClassPerformance(n_to_show=2, show_only='best')
    # Act
    result = check.run(coco_train_visiondata, coco_test_visiondata,
                       mock_trained_yolov5_object_detection, device=device)

    # Assert
    assert_that(len(result.value), equal_to(8))

    first_row = \
        result.value.loc[result.value['Metric'] == 'Average Precision'].sort_values(by='Value', ascending=False).iloc[0]
    assert_that(first_row['Value'], close_to(0.999, 0.001))
    assert_that(first_row['Number of samples'], equal_to(1))
    assert_that(first_row['Class'], is_in([28]))

    first_row = \
        result.value.loc[result.value['Metric'] == 'Average Recall'].sort_values(by='Value', ascending=False).iloc[0]
    assert_that(first_row['Value'], close_to(1, 0.001))
    assert_that(first_row['Number of samples'], equal_to(1))
    assert_that(first_row['Class'], is_in([28]))


def test_class_list(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist, device):
    # Arrange

    check = ClassPerformance(class_list_to_show=[1])
    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    # Assert
    assert_that(len(result.value), equal_to(4))
    assert_that(result.value['Class'].iloc[0], equal_to(1))
    assert_that(result.value['Class'].iloc[1], equal_to(1))
    assert_that(result.value['Class'].iloc[2], equal_to(1))
    assert_that(result.value['Class'].iloc[3], equal_to(1))


def test_condition_test_performance_not_less_than_pass(mnist_dataset_train,
                                                       mnist_dataset_test,
                                                       mock_trained_mnist,
                                                       device):
    # Arrange
    check = ClassPerformance(class_list_to_show=[1]).add_condition_test_performance_greater_than(0.5)

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Found minimum score for Precision metric of value 0.98 for class 1',
                               name='Scores are greater than 0.5'))
    )


def test_condition_test_performance_not_less_than_fail(mnist_dataset_train,
                                                       mnist_dataset_test,
                                                       mock_trained_mnist,
                                                       device):
    # Arrange
    check = ClassPerformance(n_to_show=2,
                             alternative_metrics={'p': Precision(), 'r': Recall()}) \
        .add_condition_test_performance_greater_than(1)

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details="Found metrics with scores below threshold:\n["
                                       "{'Class': '2', 'Metric': 'r', 'Score': '0.99'}, "
                                       "{'Class': '1', 'Metric': 'r', 'Score': '0.99'}, "
                                       "{'Class': '2', 'Metric': 'p', 'Score': '0.98'}, "
                                       "{'Class': '1', 'Metric': 'p', 'Score': '0.98'}]",
                               name='Scores are greater than 1'))
    )


def test_condition_train_test_relative_degradation_not_greater_than_pass(mnist_dataset_train,
                                                                         mnist_dataset_test,
                                                                         mock_trained_mnist,
                                                                         device):
    # Arrange
    check = ClassPerformance(class_list_to_show=[1]).add_condition_train_test_relative_degradation_less_than(0.1)

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Found max degradation of 0.0033% for metric Recall and class 1',
                               name='Train-Test scores relative degradation is less than 0.1'))
    )


def test_condition_train_test_relative_degradation_not_greater_than_fail(mnist_dataset_train,
                                                                         mnist_dataset_test,
                                                                         mock_trained_mnist,
                                                                         device):
    # Arrange
    check = ClassPerformance() \
        .add_condition_train_test_relative_degradation_less_than(0.0001)

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details='10 classes scores failed. Found max degradation of 0.86% for metric Precision '
                                       'and class 5',
                               name='Train-Test scores relative degradation is less than 0.0001'))
    )


def test_condition_class_performance_imbalance_ratio_not_greater_than(mnist_dataset_train,
                                                                      mnist_dataset_test,
                                                                      mock_trained_mnist,
                                                                      device):
    # Arrange
    check = ClassPerformance() \
        .add_condition_class_performance_imbalance_ratio_less_than(0.5, 'Precision')

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=True,
                               details='Relative ratio difference between highest and lowest in Test dataset classes '
                                       'is 1.78%, using Precision metric. Lowest class - 7: 0.97; Highest class - 8: '
                                       '0.99\nRelative ratio difference between highest and lowest in Train dataset '
                                       'classes is 3.78%, using Precision metric. Lowest class - 9: 0.96; Highest class'
                                       ' - 6: 0.99',
                               name='Relative ratio difference between labels \'Precision\' score is less than 50%'))
    )


def test_condition_class_performance_imbalance_ratio_not_greater_than_fail(mnist_dataset_train,
                                                                           mnist_dataset_test,
                                                                           mock_trained_mnist,
                                                                           device):
    # Arrange
    check = ClassPerformance() \
        .add_condition_class_performance_imbalance_ratio_less_than(0.0001, 'Precision')

    # Act
    result = check.run(mnist_dataset_train, mnist_dataset_test, mock_trained_mnist,
                       device=device)

    assert_that(result.conditions_results, has_items(
        equal_condition_result(is_pass=False,
                               details='Relative ratio difference between highest and lowest in Test dataset classes '
                                       'is 1.78%, using Precision metric. Lowest class - 7: 0.97; Highest class - 8: '
                                       '0.99\nRelative ratio difference between highest and lowest in Train dataset '
                                       'classes is 3.78%, using Precision metric. Lowest class - 9: 0.96; Highest class'
                                       ' - 6: 0.99',
                               name='Relative ratio difference between labels \'Precision\' score is less than 0.01%'))
    )


def test_custom_task(mnist_train_custom_task, mnist_test_custom_task, device, mock_trained_mnist):
    # Arrange
    metrics = {'metric': Precision()}
    check = ClassPerformance(alternative_metrics=metrics)

    # Act & Assert - check runs without errors
    check.run(mnist_train_custom_task, mnist_test_custom_task, model=mock_trained_mnist, device=device)
