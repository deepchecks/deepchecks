import numpy as np
import pandas as pd
from hamcrest import *

from deepchecks.core.check_result import CheckResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.checks.data_integrity import PercentOfNulls
from tests.base.utils import equal_condition_result


def test_percent_of_nulls():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls().run(df)
    # Assert
    assert_that(
        result,
        all_of(
            instance_of(CheckResult),
            has_property('value', has_entries({
                'foo': equal_to(0.5),
                'bar': equal_to(0.25),
            }))
        )
    )


def test_percent_of_nulls_with_columns_of_categorical_dtype():
    # Arrange
    t = pd.CategoricalDtype(categories=['b', 'a'])
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']}, dtype=t)
    # Act
    result = PercentOfNulls().run(df)
    # Assert
    assert_that(
        result,
        all_of(
            instance_of(CheckResult),
            has_property('value', has_entries({
                'foo': equal_to(0.5),
                'bar': equal_to(0.25),
            }))
        )
    )


def test_reduce_output_method():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls().run(df)
    # Assert
    assert_that(
        result.check.reduce_output(result),
        has_entries({'foo': equal_to(0.5), 'bar': equal_to(0.25)})
    )


def test_exclude_parameter():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(exclude=['foo']).run(df)
    # Assert
    assert_that(result, all_of(
        instance_of(CheckResult),
        has_property('value', has_entries({'bar': equal_to(0.25)}))
    ))


def test_columns_parameter():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    # Act
    result = PercentOfNulls(columns=['foo']).run(df)
    # Assert
    assert_that(result, all_of(
        instance_of(CheckResult),
        has_property('value', has_entries({'foo': equal_to(0.5)}))
    ))


def test_condition():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b'], 'bar': ['a', 'a']})
    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01)
    # Act
    check_result = check.run(df)
    conditions_results = check.conditions_decision(check_result)

    assert_that(conditions_results, contains_exactly(
        equal_condition_result(
            is_pass=True,
            details='',
            name='Percent of null values in each column is not greater than 1%',
        )
    ))


def test_not_passing_condition():
    # Arrange
    df = pd.DataFrame({'foo': ['a','b', np.nan, None], 'bar': [None, 'a', 'b', 'a']})
    check = PercentOfNulls().add_condition_percent_of_nulls_not_greater_than(0.01)
    # Act
    check_result = check.run(df)
    conditions_results = check.conditions_decision(check_result)

    assert_that(conditions_results, contains_exactly(
        equal_condition_result(
            is_pass=False,
            details='Columns with percent of null values greater than 1%: foo, bar',
            name='Percent of null values in each column is not greater than 1%',
        )
    ))
