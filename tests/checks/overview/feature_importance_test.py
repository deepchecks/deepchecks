"""Tests for Feature Importance."""
import pandas as pd

from mlchecks.checks.overview.feature_importance import feature_importance, FeatureImportance
from mlchecks.base import Dataset
from mlchecks.utils import MLChecksValueError
from hamcrest import assert_that, calling, raises

def test_feature_importance_function(iris_random_forest, iris_dataset_labeled):
    # Act
    result = feature_importance(iris_dataset_labeled, iris_random_forest)

    # Assert
    assert result.value

def test_feature_importance_not_binary(iris_random_forest, iris_dataset_labeled):
    # Arrange
    result = feature_importance(iris_dataset_labeled, iris_random_forest, plot_type='beeswarm')
    # Act & Assert
    assert_that(
        calling(result._ipython_display_).with_args(),
        raises(MLChecksValueError, 'Only plot_type = \'bar\' is supported for multi-class models</p>'))


def test_feature_importance_binary(iris_random_forest_single_class, iris_dataset_single_class_labeled):
    # Act
    result = feature_importance(iris_dataset_single_class_labeled,
                                iris_random_forest_single_class,
                                plot_type='beeswarm')

    # Assert
    assert result.value


def test_feature_importance_object(iris_random_forest, iris_dataset_labeled):
    # Arrange
    suit_runner = FeatureImportance()
    # Act
    result = suit_runner.run(iris_dataset_labeled, iris_random_forest)
    # Assert
    assert result.value is not None


def test_feature_importance_unsuported_model(iris_adaboost, iris_dataset_labeled):
    # Act
    result = feature_importance(iris_dataset_labeled, iris_adaboost)
    # Assert
    assert result.value is None

def test_feature_importance_bad_plot(iris_random_forest, iris_dataset_labeled):
    # Arrange
    result = feature_importance(iris_dataset_labeled, iris_random_forest, plot_type='bad_plot')
    # Act & Assert
    assert_that(
        calling(result._ipython_display_).with_args(),
        raises(MLChecksValueError, 'plot_type=\'bad_plot\' currently not supported. Use \'beeswarm\' or \'bar\''))

def test_feature_importance_unmatching_dataset(iris_random_forest):
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['Nan', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    dataset = Dataset(dataframe, label='col3')

    # Assert
    assert_that(
        calling(feature_importance).with_args(dataset, iris_random_forest),
        raises(MLChecksValueError))
