"""Contains unit tests for the category_mismatch_train_validation check"""

import pandas as pd
from mlchecks.base import Dataset
from mlchecks.utils import MLChecksValueError
from mlchecks.checks.integrity import category_mismatch_train_validation, CategoryMismatchTrainValidation
from hamcrest import assert_that, calling, raises, equal_to, has_length


def test_dataset_wrong_input():
    x = 'wrong_input'
    # Act & Assert
    assert_that(calling(category_mismatch_train_validation).with_args(x, x),
                raises(MLChecksValueError,
                'dataset must be of type DataFrame or Dataset. instead got: str'))


def test_no_mismatch():
    train_data = {'col1': ['a', 'b', 'c']}
    validation_data = {'col1': ['a', 'a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainValidation()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_length(0))


def test_new_category():
    train_data = {'col1': ['a', 'b', 'c']}
    validation_data = {'col1': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainValidation()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['new categories'][0], equal_to(set('d')))
    assert_that(result['missing categories'][0], equal_to(None))
    assert_that(result['shared categories'][0], equal_to(set(['a', 'b', 'c'])))


def test_missing_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainValidation()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['new categories'][0], equal_to(None))
    assert_that(result['missing categories'][0], equal_to(set('d')))
    assert_that(result['shared categories'][0], equal_to(set(['a', 'b', 'c'])))


def test_missing_new_category():
    train_data = {'col1': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c', 'e']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainValidation()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['new categories'][0], equal_to(set('e')))
    assert_that(result['missing categories'][0], equal_to(set('d')))
    assert_that(result['shared categories'][0], equal_to(set(['a', 'b', 'c'])))


def test_multiple_categories():
    train_data = {'col1': ['a', 'b', 'c', 'd'], 'col2': ['a', 'b', 'c', 'd']}
    validation_data = {'col1': ['a', 'b', 'c', 'e'], 'col2': ['a', 'b', 'c', 'd']}
    train_dataset = Dataset(pd.DataFrame(data=train_data, columns=['col1']), cat_features=['col1'])
    validation_dataset = Dataset(pd.DataFrame(data=validation_data, columns=['col1']), cat_features=['col1'])

    # Arrange
    check = CategoryMismatchTrainValidation()
    # Act X
    result = check.run(validation_dataset=validation_dataset, train_dataset=train_dataset).value
    # Assert
    assert_that(result, has_length(1))
    assert_that(result['new categories'][0], equal_to(set('e')))
    assert_that(result['missing categories'][0], equal_to(set('d')))
    assert_that(result['shared categories'][0], equal_to(set(['a', 'b', 'c'])))
