"""Unused features tests."""
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

from deepchecks import Dataset
from deepchecks.checks.methodology import UnusedFeatures
from hamcrest import assert_that, equal_to


def test_unused_feature_detection(iris_split_dataset_and_model_rf):
    # Arrange
    train, validation, clf = iris_split_dataset_and_model_rf

    # Act
    result = UnusedFeatures().run(train, validation, clf)

    # Assert
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)',
                                                              'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to({'sepal width (cm)'}))
    assert_that(set(result.value['unused features']['low variance']), equal_to(set()))


def test_low_feature_importance_threshold(iris_split_dataset_and_model_rf):
    # Arrange
    train, validation, clf = iris_split_dataset_and_model_rf

    # Act
    result = UnusedFeatures(feature_importance_threshold=0).run(train, validation, clf)

    # Assert
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)',
                                                              'sepal width (cm)', 'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to(set()))
    assert_that(set(result.value['unused features']['low variance']), equal_to(set()))


def test_higher_variance_threshold(iris_split_dataset_and_model_rf):
    # Arrange
    train, validation, clf = iris_split_dataset_and_model_rf

    # Act
    result = UnusedFeatures(feature_variance_threshold=2).run(train, validation, clf)

    # Assert
    assert_that(set(result.value['used features']), equal_to({'petal width (cm)', 'petal length (cm)',
                                                              'sepal length (cm)'}))
    assert_that(set(result.value['unused features']['high variance']), equal_to(set()))
    assert_that(set(result.value['unused features']['low variance']), equal_to({'sepal width (cm)'}))
