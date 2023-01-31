# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains unit tests for the new_label_train_validation check"""

import pandas as pd
from hamcrest import assert_that, calling, equal_to, greater_than, has_items, has_length, raises

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.train_test_validation import NewLabelTrainTest
from deepchecks.tabular.dataset import Dataset
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(NewLabelTrainTest().run).with_args(x, x),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str'
        )
    )


def test_no_new_label():
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 1, 2, 3]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({}))


def test_new_label():
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset)
    # Assert
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(1))
    assert_that(result.value['n_samples'], equal_to(4))
    assert_that(result.value['new_labels'], equal_to([4]))
    assert_that(result.display, has_length(greater_than(0)))


def test_new_label_without_display():
    train_data = {'col1': [1, 2, 3]}
    test_data = {'col1': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset, with_display=False)
    # Assert
    assert_that(result.value)
    assert_that(result.value['n_new_labels_samples'], equal_to(1))
    assert_that(result.value['n_samples'], equal_to(4))
    assert_that(result.value['new_labels'], equal_to([4]))
    assert_that(result.display, has_length(0))


def test_missing_label():
    train_data = {'col1': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result, equal_to({}))


def test_missing_new_label():
    train_data = {'col1': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result)
    assert_that(result['n_new_labels_samples'], equal_to(1))
    assert_that(result['n_samples'], equal_to(4))
    assert_that(result['new_labels'], equal_to([5]))


def test_multiple_categories():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result)
    assert_that(result['n_new_labels_samples'], equal_to(1))
    assert_that(result['n_samples'], equal_to(4))
    assert_that(result['new_labels'], equal_to([5]))


def test_condition_number_of_new_labels_pass():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found 1 new labels in test data: [5]',
                               name='Number of new label values is less or equal to 3')
    ))


def test_condition_number_of_new_labels_fail():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_labels_number_less_or_equal(0)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 new labels in test data: [5]',
                               name='Number of new label values is less or equal to 0')
    ))


def test_condition_ratio_of_new_label_samples_pass():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_label_ratio_less_or_equal(0.3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found 25% of labels in test data are new labels: [5]',
                               name='Ratio of samples with new label is less or equal to 30%')
    ))


def test_condition_ratio_of_new_label_samples_fail():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="multiclass")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="multiclass")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_label_ratio_less_or_equal(0.1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 25% of labels in test data are new labels: [5]',
                               name='Ratio of samples with new label is less or equal to 10%')
    ))
