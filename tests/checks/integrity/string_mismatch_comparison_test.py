"""Contains unit tests for the string_mismatch check."""
import pandas as pd

from mlchecks.checks import StringMismatchComparison

from hamcrest import assert_that, has_length, has_entry, has_entries, has_items

from tests.checks.utils import equal_condition_result


def test_single_col_mismatch():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 1 column, 1 baseform are mismatch
    assert_that(result, has_entry('col1', has_length(1)))


def test_mismatch_multi_column():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?'],
            'col2': ['aaa', 'bbb', 'ddd', '><', '123', '111', '444']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep'],
                     'col2': ['aaa!', 'bbb!', 'ddd', '><', '123???', '123!', '__123__', '111']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert - 2 columns, 4 baseforms are mismatch
    assert_that(result, has_entries({
        'col1': has_length(1), 'col2': has_length(3)
     }))


def test_no_mismatch():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat']}
    compared_data = {'col1': ['foo', 'foo', 'bar', 'bar', 'bar', 'dog?!']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_column():
    # Arrange
    data = {'col1': ['foo', 'bar', 'cat'], 'col2': [10, 2.3, 1]}
    compared_data = {'col1': ['foo', 'foo', 'foo'], 'col2': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_numeric_string_column():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': ['1', '2.30', '1.0']}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_no_mismatch_on_one_column_numeric():
    # Arrange
    data = {'num_str': ['10', '2.3', '1']}
    compared_data = {'num_str': [1, 2.30, 1.0]}

    # Act
    result = StringMismatchComparison().run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)).value

    # Assert
    assert_that(result, has_length(0))


def test_condition_no_new_variants_fail():
    # Arrange
    data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    compared_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_no_new_variants()
    # Act
    result = check.conditions_decision(check.run(pd.DataFrame(data=data), pd.DataFrame(data=compared_data)))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='No new variants allowed in validation data',
                               details='Found columns with variants: [\'col1\']')
    ))


def test_condition_no_new_variants_pass():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', 'cat', 'earth', 'foo', 'bar', 'foo?', 'bar']}
    check = StringMismatchComparison().add_condition_no_new_variants()
    # Act
    result = check.conditions_decision(check.run(pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='No new variants allowed in validation data')
    ))


def test_condition_percent_new_variants_fail():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_percent_new_variants_no_more_than(0.1)
    # Act
    result = check.conditions_decision(check.run(pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='No more than 10.00% new variants in validation data',
                               details='Found columns with variants: [\'col1\']')
    ))


def test_condition_percent_new_variants_pass():
    # Arrange
    base_data = {'col1': ['Deep', 'deep', 'deep!!!', 'earth', 'foo', 'bar', 'foo?']}
    tested_data = {'col1': ['Deep', 'deep', '$deeP$', 'earth', 'foo', 'bar', 'foo?', '?deep']}
    check = StringMismatchComparison().add_condition_percent_new_variants_no_more_than(0.5)
    # Act
    result = check.conditions_decision(check.run(pd.DataFrame(data=tested_data), pd.DataFrame(data=base_data)))
    # Assert
    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='No more than 50.00% new variants in validation data')
    ))
