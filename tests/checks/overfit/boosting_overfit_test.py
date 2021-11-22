"""Boosting overfit tests."""
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split

from deepchecks import Dataset
from deepchecks.checks.overfit.boosting_overfit import BoostingOverfit
from hamcrest import assert_that, close_to


def test_boosting_classifier(iris):
    # Arrange
    train_df, validation_df = train_test_split(iris, test_size=0.33, random_state=0)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingClassifier(random_state=0)
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = BoostingOverfit().run(train, validation, clf)

    # Assert
    assert_that(result.value, close_to(0.92, 0.05))


def test_boosting_regressor(diabetes, diabetes_model):
    # Arrange
    train, validation = diabetes

    # Act
    result = BoostingOverfit().run(train, validation, diabetes_model)

    # Assert
    assert_that(result.value, close_to(-57, 5))


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
    assert_that(result.value, close_to(0.95, 0.05))
