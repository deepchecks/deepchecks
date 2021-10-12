from mlchecks.checks.overview.feature_importance import *
from hamcrest import *

def assert_result(result):
    pass

def test_feature_importance_function(iris_random_forest, iris_dataset ):
    # Act
    result = feature_importance(iris_dataset, iris_random_forest)

    # Assert
    assert(result.value)

def test_feature_importance_not_binary(iris_random_forest, iris_dataset ):
    # Act
    result = feature_importance(iris_dataset, iris_random_forest, plot_type='beeswarm')

    # Assert
    assert result.value is None


def test_feature_importance_not_binary(iris_random_forest_single_class, iris_dataset_single_class):
    # Act
    result = feature_importance(iris_dataset_single_class, iris_random_forest_single_class,plot_type='beeswarm')

    # Assert
    assert result.value is not None


def test_feature_importance_object(iris_random_forest, iris_dataset):
    # Arrange
    suit_runner = FeatureImportance()
    # Act
    result = suit_runner.run(iris_dataset, iris_random_forest)
    # Assert
    assert_result(result)


def test_feature_importance_unsuported_model(iris_adaboost, iris_dataset):
    # Act
    result = feature_importance(iris_dataset, iris_adaboost)
    # Assert
    assert result.value is None
