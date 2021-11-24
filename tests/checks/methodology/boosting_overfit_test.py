"""Boosting overfit tests."""
from statistics import mean

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from deepchecks import Dataset
from deepchecks.checks.methodology.boosting_overfit import BoostingOverfit
from hamcrest import assert_that, close_to, has_length

from tests.checks.utils import equal_condition_result


def test_boosting_classifier(iris):
    # Arrange
    train_df, test = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    test = Dataset(test, label='target')

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = BoostingOverfit().run(train, test, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.999, 0.01))
    assert_that(mean(test_scores), close_to(0.961, 0.01))


def test_boosting_regressor(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes

    # Act
    result = BoostingOverfit().run(train, validation, diabetes_model)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(-44.52, 0.01))
    assert_that(mean(test_scores), close_to(-59.35, 0.01))


def test_boosting_classifier_with_metric(iris):
    # Arrange
    train_df, validation_df = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = BoostingOverfit(metric='recall_micro').run(train, validation, clf)

    # Assert
    train_scores = result.value['train']
    test_scores = result.value['test']
    assert_that(train_scores, has_length(20))
    assert_that(test_scores, has_length(20))
    assert_that(mean(train_scores), close_to(0.999, 0.01))
    assert_that(mean(test_scores), close_to(0.96, 0.01))


def test_condition_score_decline_not_greater_than_pass(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes
    check = BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than()

    # Act
    condition_result, *_ = check.conditions_decision(check.run(train, validation, diabetes_model))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        name='Test score decline is not greater than 5.00%'
    ))


def test_condition_score_percentage_decline_not_greater_than_not_pass(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes
    check = BoostingOverfit().add_condition_test_score_percent_decline_not_greater_than(0.01)

    # Act
    condition_result, *_ = check.conditions_decision(check.run(train, validation, diabetes_model))

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Test score decline is not greater than 1.00%',
        details='Found metric decline of: -3.64%'
    ))
