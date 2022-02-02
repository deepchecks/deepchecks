# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
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
from hamcrest import assert_that, calling, raises, equal_to, has_items

from deepchecks.tabular import Dataset
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.integrity import NewLabelTrainTest

from tests.checks.utils import equal_condition_result


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
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="classification_label")

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
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="classification_label")

    # Arrange
    check = NewLabelTrainTest()
    # Act X
    result = check.run(train_dataset=train_dataset, test_dataset=test_dataset).value
    # Assert
    assert_that(result)
    assert_that(result['n_new_labels_samples'], equal_to(1))
    assert_that(result['n_samples'], equal_to(4))
    assert_that(result['new_labels'], equal_to([4]))


def test_missing_label():
    train_data = {'col1': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']),
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="classification_label")

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
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1']),
                           label='col1', label_type="classification_label")

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
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="classification_label")

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
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="classification_label")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_labels_not_greater_than(3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Number of new label values is not greater than 3')
    ))


def test_condition_number_of_new_labels_fail():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="classification_label")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_labels_not_greater_than(0)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found 1 new labels: [5]',
                               name='Number of new label values is not greater than 0')
    ))


def test_condition_ratio_of_new_label_samples_pass():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="classification_label")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_label_ratio_not_greater_than(0.3)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='Ratio of samples with new label is not greater than 30%')
    ))


def test_condition_ratio_of_new_label_samples_fail():
    train_data = {'col1': [1, 2, 3, 4], 'col2': [1, 2, 3, 4]}
    test_data = {'col1': [1, 2, 3, 5], 'col2': [1, 2, 3, 4]}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']),
                            label='col1', label_type="classification_label")
    test_dataset = Dataset(pd.DataFrame(data=test_data, columns=['col1', 'col2']),
                           label='col1', label_type="classification_label")

    # Arrange
    check = NewLabelTrainTest().add_condition_new_label_ratio_not_greater_than(0.1)

    # Act
    result = check.conditions_decision(check.run(train_dataset, test_dataset))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found new labels in test data: [5]\nmaking 25% of samples.',
                               name='Ratio of samples with new label is not greater than 10%')
    ))
