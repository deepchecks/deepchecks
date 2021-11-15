"""Tests for Feature Importance."""
import numpy as np
import pandas as pd

from deepchecks.checks.overview.feature_importance import FeatureImportance
from deepchecks.base import Dataset
from deepchecks.utils import DeepchecksValueError
from hamcrest import assert_that, calling, raises


def test_feature_importance_function(iris_random_forest, iris_labeled_dataset):
    # Act
    result = FeatureImportance().run(iris_labeled_dataset, iris_random_forest)

    # Assert
    assert result.value


def test_feature_importance_not_binary(iris_random_forest, iris_labeled_dataset):
    # Arrange
    result = FeatureImportance(plot_type='beeswarm').run(iris_labeled_dataset, iris_random_forest)
    # Act & Assert
    assert_that(
        # pylint: disable=protected-access
        calling(result._ipython_display_).with_args(),
        raises(DeepchecksValueError, 'Only plot_type = \'bar\' is supported for multi-class models</p>'))


def test_feature_importance_binary(iris_random_forest_single_class, iris_dataset_single_class_labeled):
    # Act
    result = FeatureImportance(plot_type='beeswarm').run(iris_dataset_single_class_labeled,
                                                         iris_random_forest_single_class, )

    # Assert
    assert result.value


def test_feature_importance_object(iris_random_forest, iris_labeled_dataset):
    # Arrange
    suit_runner = FeatureImportance()
    # Act
    result = suit_runner.run(iris_labeled_dataset, iris_random_forest)
    # Assert
    assert result.value is not None


def test_feature_importance_unsuported_model(iris_adaboost, iris_labeled_dataset):
    # Act
    result = FeatureImportance().run(iris_labeled_dataset, iris_adaboost)
    # Assert
    assert result.value is None


def test_feature_importance_bad_plot(iris_random_forest, iris_labeled_dataset):
    # Arrange
    result = FeatureImportance(plot_type='bad_plot').run(iris_labeled_dataset, iris_random_forest)
    # Act & Assert
    assert_that(
        # pylint: disable=protected-access
        calling(result._ipython_display_).with_args(),
        raises(DeepchecksValueError, 'plot_type=\'bad_plot\' currently not supported. Use \'beeswarm\' or \'bar\''))


def test_feature_importance_unmatching_dataset(iris_random_forest):
    data = {'col1': ['foo', 'bar', 'null'], 'col2': ['Nan', 'bar', 1], 'col3': [1, 2, 3]}
    dataframe = pd.DataFrame(data=data)
    dataset = Dataset(dataframe, label='col3')

    # Assert
    assert_that(
        calling(FeatureImportance().run).with_args(dataset, iris_random_forest),
        raises(DeepchecksValueError))


def test_nan(iris_random_forest, iris):
    df = iris.append(pd.DataFrame({'sepal length (cm)': [np.nan],
                                   'sepal width (cm)':[np.nan],
                                   'petal length (cm)':[np.nan],
                                   'petal width (cm)': [np.nan],
                                   'target':[0]}))
    # Act
    result = FeatureImportance().run(Dataset(df, label='target'), iris_random_forest)

    # Assert
    assert result.value
