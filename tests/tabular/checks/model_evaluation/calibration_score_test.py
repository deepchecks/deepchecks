# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Contains unit tests for the calibration_metric check."""
from hamcrest import assert_that, calling, close_to, greater_than, has_entries, has_length, raises

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.model_evaluation import CalibrationScore
from deepchecks.tabular.dataset import Dataset


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(CalibrationScore().run).with_args(bad_dataset, None),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label(iris, iris_adaboost):
    # Arrange
    iris = iris.drop('target', axis=1)
    ds = Dataset(iris)
    # Assert
    assert_that(
        calling(CalibrationScore().run).with_args(ds, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(
        calling(CalibrationScore().run).with_args(train, clf),
        raises(ModelValidationError, 'Check is irrelevant for regression tasks'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = CalibrationScore()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost)
    # Assert
    assert_that(result.value, has_length(3))  # iris has 3 targets

    assert_that(result.value, has_entries({
        0: close_to(0.0, 0.0001),
        1: close_to(0.026, 0.001),
        2: close_to(0.026, 0.001)
    }))
    assert_that(result.display, has_length(greater_than(0)))


def test_model_info_object_without_display(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = CalibrationScore()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost, with_display=False)
    # Assert
    assert_that(result.value, has_length(3))  # iris has 3 targets

    assert_that(result.value, has_entries({
        0: close_to(0.0, 0.0001),
        1: close_to(0.026, 0.001),
        2: close_to(0.026, 0.001)
    }))
    assert_that(result.display, has_length(0))


def test_binary_model_info_object(iris_dataset_single_class_labeled, iris_random_forest_single_class):
    # Arrange
    check = CalibrationScore()
    # Act X
    result = check.run(iris_dataset_single_class_labeled, iris_random_forest_single_class).value
    # Assert
    assert_that(result, has_length(1))

    assert_that(result, has_entries({
        0: close_to(0.0002, 0.0005)
    }))


def test_binary_string_model_info_object(iris_binary_string_split_dataset_and_model):
    # Arrange
    _, test_ds, clf = iris_binary_string_split_dataset_and_model
    check = CalibrationScore()
    # Act X
    result = check.run(test_ds, clf).value
    # Assert
    assert_that(result, has_length(1))

    assert_that(result, has_entries({
        0: close_to(0.04, 0.001)
    }))
