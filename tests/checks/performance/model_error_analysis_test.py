# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Tests for segment performance check."""
from hamcrest import assert_that, has_entries, close_to, has_property, equal_to, calling, raises, is_, has_length

from deepchecks.errors import DeepchecksValueError, DeepchecksProcessError
from deepchecks.checks.performance.model_error_analysis import ModelErrorAnalysis


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(bad_dataset, None),
                raises(DeepchecksValueError,
                       'Check requires dataset to be of type Dataset. instead got: str'))


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(iris_dataset, iris_adaboost),
                raises(DeepchecksValueError, 'Check requires dataset to have a label column'))


def test_model_error_analysis_classification(iris_labeled_dataset, iris_adaboost):
    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(iris_labeled_dataset, iris_adaboost),
                raises(DeepchecksProcessError,
                       'Unable to train meaningful error model'))


def test_model_error_analysis_regression_not_meaningful(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(val, model),
                raises(DeepchecksProcessError,
                       'Unable to train meaningful error model'))


def test_model_error_analysis_not_meaningful(diabetes_split_dataset_and_model):
    # Arrange
    _, val, model = diabetes_split_dataset_and_model

    # Assert
    assert_that(calling(ModelErrorAnalysis().run).with_args(val, model),
                raises(DeepchecksProcessError,
                       'Unable to train meaningful error model'))

