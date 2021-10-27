"""Contains unit tests for the new_label_train_validation check"""

import pandas as pd
from mlchecks.base import Dataset
from mlchecks.utils import MLChecksValueError
from mlchecks.checks.integrity import new_label_train_validation, NewLabelTrainValidation
from hamcrest import assert_that, calling, raises, has_length, close_to, equal_to


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(new_label_train_validation).with_args(x, x),
                raises(MLChecksValueError,
                'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_no_new_label():
    train_data = {'col1': ['a', 'b', 'c']}
    validation_data = {'col1': ['a', 'a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1')
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), label='col1')

    # Arrange
    check = NewLabelTrainValidation()
    # Act X
    result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset).value
    # Assert
    assert_that(result, equal_to({}))


def test_new_label():
    train_data = {'col1': ['a', 'b', 'c']}
    validation_data = {'col1': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1')
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), label='col1')

    # Arrange
    check = NewLabelTrainValidation()
    # Act X
    result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['col1'], close_to(0.25, 0.01))


def test_missing_label():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1')
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), label='col1')

    # Arrange
    check = NewLabelTrainValidation()
    # Act X
    result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset).value
    # Assert
    assert_that(result, equal_to({}))


def test_missing_new_label():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c', 'e']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), label='col1')
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), label='col1')

    # Arrange
    check = NewLabelTrainValidation()
    # Act X
    result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['col1'], close_to(0.25, 0.01))


def test_multiple_categories():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1', 'col2']), label='col1')
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1', 'col2']), label='col1')

    # Arrange
    check = NewLabelTrainValidation()
    # Act X
    result = check.run(train_dataset=train_dataset, validation_dataset=validation_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['col1'], close_to(0.25, 0.01))
