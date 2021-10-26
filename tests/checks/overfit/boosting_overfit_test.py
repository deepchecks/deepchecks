"""Boosting overfit tests."""
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

from mlchecks import Dataset
from mlchecks.checks.overfit.boosting_overfit import boosting_overfit
from hamcrest import assert_that, close_to


def test_boosting_classifier(iris):
    # Arrange
    train_df, validation_df = train_test_split(iris, test_size=0.33)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingClassifier()
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = boosting_overfit(train, validation, clf)

    # Assert
    assert_that(result.value, close_to(0.93, 0.05))


def test_boosting_regressor(diabetes):
    # Arrange
    train_df, validation_df = train_test_split(diabetes, test_size=0.33)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingRegressor()
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = boosting_overfit(train, validation, clf)

    # Assert
    assert_that(result.value, close_to(57, 5))


def test_boosting_classifier_with_metric(iris):
    # Arrange
    train_df, validation_df = train_test_split(iris, test_size=0.33)
    train = Dataset(train_df, label='target')
    validation = Dataset(validation_df, label='target')

    clf = GradientBoostingClassifier()
    clf.fit(train.features_columns(), train.label_col())

    # Act
    result = boosting_overfit(train, validation, clf, metric='recall_micro')

    # Assert
    assert_that(result.value, close_to(0.95, 0.05))
