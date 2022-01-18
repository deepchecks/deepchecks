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
"""Contains unit tests for the confusion_matrix_report check."""
import numpy as np
from hamcrest import assert_that, calling, raises
from deepchecks.checks.performance import ConfusionMatrixReport
from deepchecks.errors import DeepchecksValueError, DatasetValidationError, ModelValidationError


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(bad_dataset, None),
        raises(
            DeepchecksValueError,
            'non-empty Dataset instance was expected, instead got str')
    )


def test_dataset_no_label(iris_dataset, iris_adaboost):
    # Assert
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(iris_dataset, iris_adaboost),
        raises(DatasetValidationError, 'Check is irrelevant for Datasets without label')
    )


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(train, clf),
        raises(
            ModelValidationError,
            r'Check is relevant for models of type \[\'multiclass\', \'binary\'\], '
            r'but received model of type \'regression\'')
    )


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value
    # Assert
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert isinstance(result[i][j] , np.int64)
