import pytest
from hamcrest import assert_that, has_length
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from checks.performance import MultiModelPerformanceReport


@pytest.fixture
def classification_models(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    model2 = RandomForestClassifier(random_state=0)
    model2.fit(train.features_columns, train.label_col)
    model3 = DecisionTreeClassifier(random_state=0)
    model3.fit(train.features_columns, train.label_col)
    return train, test, model, model2, model3


@pytest.fixture
def regression_models(diabetes_split_dataset_and_model):
    train, test, model = diabetes_split_dataset_and_model
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(train.features_columns, train.label_col)
    model3 = DecisionTreeRegressor(random_state=0)
    model3.fit(train.features_columns, train.label_col)
    return train, test, model, model2, model3


def test_multi_classification(classification_models):
    # Arrange
    train, test, model, model2, model3 = classification_models
    # Act
    result = MultiModelPerformanceReport().run(train, test, [model, model2, model3])
    # Assert - 3 classes X 3 metrics X 3 models
    assert_that(result.value, has_length(27))


def test_regression(regression_models):
    # Arrange
    train, test, model, model2, model3 = regression_models
    # Act
    result = MultiModelPerformanceReport().run(train, test, [model, model2, model3])
    # Assert - 2 metrics X 3 models
    assert_that(result.value, has_length(6))
