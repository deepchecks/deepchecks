# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
import pytest
from hamcrest import assert_that, has_length
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from deepchecks.tabular.checks import MultiModelPerformanceReport


@pytest.fixture
def classification_models(iris_split_dataset_and_model):
    train, test, model = iris_split_dataset_and_model
    model2 = RandomForestClassifier(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeClassifier(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
    return train, test, model, model2, model3


@pytest.fixture
def regression_models(diabetes_split_dataset_and_model):
    train, test, model = diabetes_split_dataset_and_model
    model2 = RandomForestRegressor(random_state=0)
    model2.fit(train.data[train.features], train.data[train.label_name])
    model3 = DecisionTreeRegressor(random_state=0)
    model3.fit(train.data[train.features], train.data[train.label_name])
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
    # Assert - 3 metrics X 3 models
    assert_that(result.value, has_length(9))
