# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""Unused features tests."""
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
