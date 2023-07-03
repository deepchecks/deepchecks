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
"""Contains unit tests for the confusion_matrix_report check."""
import numpy as np
from hamcrest import assert_that, calling, greater_than, has_length, raises, equal_to

from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError, ModelValidationError
from deepchecks.tabular.checks.model_evaluation import ConfusionMatrixReport
from deepchecks.utils.strings import format_number, format_percent
from tests.base.utils import equal_condition_result


def test_dataset_wrong_input():
    bad_dataset = 'wrong_input'
    # Act & Assert
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(bad_dataset, None),
        raises(
            DeepchecksValueError,
            'non-empty instance of Dataset or DataFrame was expected, instead got str')
    )


def test_dataset_no_label(iris_dataset_no_label, iris_adaboost):
    # Assert
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(iris_dataset_no_label, iris_adaboost),
        raises(DeepchecksNotSupportedError,
               'Dataset does not contain a label column')
    )


def test_regresion_model(diabetes_split_dataset_and_model):
    # Assert
    train, _, clf = diabetes_split_dataset_and_model
    assert_that(
        calling(ConfusionMatrixReport().run).with_args(train, clf),
        raises(ModelValidationError, 'Check is irrelevant for regression tasks'))


def test_model_info_object(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost)
    res_val = result.value.to_numpy()
    # Assert
    for i in range(len(res_val)):
        for j in range(len(res_val[i])):
            assert isinstance(res_val[i][j], np.int64)
    assert_that(result.display, has_length(greater_than(0)))


def test_model_info_object_without_display(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport()
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost, with_display=False)
    res_val = result.value.to_numpy()
    # Assert
    for i in range(len(res_val)):
        for j in range(len(res_val[i])):
            assert isinstance(res_val[i][j], np.int64)
    assert_that(result.display, has_length(0))


def test_model_info_object_not_normalize(iris_labeled_dataset, iris_adaboost):
    # Arrange
    check = ConfusionMatrixReport(normalize_display=False)
    # Act X
    result = check.run(iris_labeled_dataset, iris_adaboost).value.to_numpy()
    # Assert
    for i in range(len(result)):
        for j in range(len(result[i])):
            assert isinstance(result[i][j], np.int64)



def test_condition_misclassified_samples_lower_than_raises_error(iris_split_dataset_and_model):
    # Arrange
    _, test, clf = iris_split_dataset_and_model

    # Act
    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=-0.1) \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=1.1)

    result = check.run(test, clf)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_number(-0.1 * 100)}% of the total samples',
        details='Exception in condition: DeepchecksValueError: Condition requires the parameter "misclassified_samples_threshold" '
                'to be between 0 and 1 inclusive but got -0.1',
        category=ConditionCategory.ERROR
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_number(1.1 * 100)}% of the total samples',
        details='Exception in condition: DeepchecksValueError: Condition requires the parameter "misclassified_samples_threshold" '
                'to be between 0 and 1 inclusive but got 1.1',
        category=ConditionCategory.ERROR
    ))


def test_condition_misclassified_samples_lower_than_passes(iris_split_dataset_and_model):
    # Arrange
    _, test, clf = iris_split_dataset_and_model

    threshold = 0.1

    # Act
    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=threshold)

    result = check.run(test, clf)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details = 'All misclassified confusion matrix cells contain less than ' \
                  f'{format_percent(threshold)} of the data.'
    ))


def test_condition_misclassified_samples_lower_than_fails(iris_split_dataset_and_model):
    # Arrange
    _, test, clf = iris_split_dataset_and_model

    threshold = 0.05
    thresh_samples = round(np.ceil(threshold * len(test)))

    # Act
    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=threshold)

    result = check.run(test, clf)

    class_names = result.value.columns
    confusion_matrix = result.value.to_numpy()
    m, n = confusion_matrix.shape[0], confusion_matrix.shape[1]

    n_cells_above_thresh = 0
    max_misclassified_cell_idx = (0, 1)

    # Looping over the confusion matrix and checking only the misclassified cells
    for i in range(m):
        for j in range(n):
            # omitting the principal axis of the confusion matrix
            if i != j:
                n_samples = confusion_matrix[i][j]

                if n_samples > thresh_samples:
                    n_cells_above_thresh += 1

                    x, y = max_misclassified_cell_idx
                    max_misclassified_samples = confusion_matrix[x][y]
                    if n_samples > max_misclassified_samples:
                        max_misclassified_cell_idx = (i, j)

    # Assert
    x, y = max_misclassified_cell_idx
    max_misclassified_samples = confusion_matrix[x][y]
    max_misclassified_samples_ratio = max_misclassified_samples / len(test)

    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details = f'Detected {n_cells_above_thresh} misclassified confusion matrix cell(s) each one ' \
                  f'containing more than {format_percent(threshold)} of the data. ' \
                  f'Largest misclassified cell ({format_percent(max_misclassified_samples_ratio)} of the data) ' \
                  f'is samples with a true value of "{class_names[x]}" and a predicted value of "{class_names[y]}".'
    ))


def test_confusion_matrix_report_display(iris_split_dataset_and_model):
    # Arrange
    _, test, clf = iris_split_dataset_and_model

    # Act
    check = ConfusionMatrixReport()

    result = check.run(test, clf)

    # Assert
    assert_that(result.display[0],
                equal_to('The overall accuracy of your model is: 91.67%.<br>Best accuracy achieved on samples with <b>'
                         '0</b> label (100.0%).<br>Worst accuracy achieved on samples with <b>2</b> label (75.0%).'))
    # # First is the text description and second is the heatmap
    assert_that(len(result.display), equal_to(2))
    assert_that(len(result.display[1].data), equal_to(1))
    assert_that(result.display[1].data[0].type, equal_to('heatmap'))