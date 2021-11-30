"""Tests for Model Info."""
from hamcrest import assert_that, has_entries, calling, raises
from deepchecks.checks.overview.model_info import ModelInfo
from deepchecks.errors import DeepchecksValueError


def assert_model_result(result):
    assert_that(result.value, has_entries(type='AdaBoostClassifier',
                                          params=has_entries(algorithm='SAMME.R',
                                                             learning_rate=1,
                                                             n_estimators=50)))


def test_model_info_function(iris_adaboost):
    # Act
    result = ModelInfo().run(iris_adaboost)

    # Assert
    assert_model_result(result)


def test_model_info_object(iris_adaboost):
    # Arrange
    mi = ModelInfo()
    # Act
    result = mi.run(iris_adaboost)
    # Assert
    assert_model_result(result)


def test_model_info_wrong_input():
    # Act
    assert_that(calling(ModelInfo().run).with_args('some string'),
                raises(DeepchecksValueError, 'Model must inherit from one of supported models:'))
