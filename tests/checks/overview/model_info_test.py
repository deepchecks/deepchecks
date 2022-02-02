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
"""Tests for Model Info."""
from hamcrest import assert_that, has_entries, calling, raises
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from deepchecks.tabular.checks.overview.model_info import ModelInfo
from deepchecks.core.errors import ModelValidationError


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


def test_model_info_pipeline(iris_adaboost):
    # Arrange
    simple_pipeline = Pipeline([('nan_handling', SimpleImputer(strategy='most_frequent')),
                                ('adaboost', iris_adaboost)])
    # Act
    result = ModelInfo().run(simple_pipeline)
    # Assert
    assert_model_result(result)


def test_model_info_wrong_input():
    # Act
    assert_that(
        calling(ModelInfo().run).with_args('some string'),
        raises(
            ModelValidationError,
            r'Model supplied does not meets the minimal interface requirements. Read more about .*')
    )
