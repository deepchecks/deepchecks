"""Contains unit tests for the RegressionErrorDistribution check."""
from hamcrest import assert_that, calling, raises, has_items, close_to

from deepchecks.base import Dataset
from deepchecks.checks.performance import RegressionErrorDistribution
from deepchecks.errors import DeepchecksValueError
from tests.checks.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(RegressionErrorDistribution().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check RegressionErrorDistribution requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(diabetes_df, diabetes_model):
    # Assert
    assert_that(calling(RegressionErrorDistribution().run).with_args(Dataset(diabetes_df), diabetes_model),
                raises(DeepchecksValueError, 'Check RegressionErrorDistribution requires dataset to have a label column'))


def test_multiclass_model(iris_split_dataset_and_model):
    # Assert
    _, test, clf = iris_split_dataset_and_model
    assert_that(calling(RegressionErrorDistribution().run).with_args(test, clf),
                raises(DeepchecksValueError, r'Check RegressionErrorDistribution Expected model to be a type from'
                                           r' \[\'regression\'\], but received model of type: multiclass'))


def test_regression_error_distribution(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution()
    # Act X
    result = check.run(test, clf).value
    # Assert
    assert_that(result['kurtosis'], close_to(0.028, 0.001))
    assert_that(result['pvalue'], close_to(0.72, 0.01))


def test_regression_error_distribution_less(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution(alternative='less')
    # Act X
    result = check.run(test, clf).value
    # Assert
    assert_that(result['kurtosis'], close_to(0.028, 0.001))
    assert_that(result['pvalue'], close_to(0.63, 0.01))


def test_regression_error_distribution_greater(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    check = RegressionErrorDistribution(alternative='greater')
    # Act X
    result = check.run(test, clf).value
    # Assert
    assert_that(result['kurtosis'], close_to(0.028, 0.001))
    assert_that(result['pvalue'], close_to(0.36, 0.01))


def test_condition_absolute_kurtosis_not_greater_than_not_passed(diabetes_split_dataset_and_model):
    # Arrange
    _, test, clf = diabetes_split_dataset_and_model
    test = Dataset(test.data.copy(), label='target')
    test._data[test.label_name] =300

    check = RegressionErrorDistribution().add_condition_p_value_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='P value not less than 0.0001',
                               details='p value: 5e-05')
    ))


def test_condition_absolute_kurtosis_not_greater_than_passed(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_p_value_not_less_than()

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=True,
                               name='P value not less than 0.0001')
    )) 

def test_condition_absolute_kurtosis_not_greater_than_not_passed_0_max(diabetes_split_dataset_and_model):
    _, test, clf = diabetes_split_dataset_and_model

    check = RegressionErrorDistribution().add_condition_p_value_not_less_than(p_value_threshold=1)

    # Act
    result = check.conditions_decision(check.run(test, clf))

    assert_that(result, has_items(
        equal_condition_result(is_pass=False,
                               name='P value not less than 1',
                               details='p value: 0.72725')
    )) 
