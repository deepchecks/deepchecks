from mlchecks.checks.overview.model_info import *
from hamcrest import *


def assert_model_result(result):
    assert_that(result.value, has_entries(type='AdaBoostClassifier',
                                          params=has_entries(algorithm='SAMME.R',
                                                             learning_rate=1,
                                                             n_estimators=50)))


def test_model_info_function(skmodel):
    # Act
    result = model_info(skmodel)

    # Assert
    assert_model_result(result)


def test_model_info_object(skmodel):
    # Arrange
    mi = ModelInfo()
    # Act
    result = mi.run(skmodel)
    # Assert
    assert_model_result(result)