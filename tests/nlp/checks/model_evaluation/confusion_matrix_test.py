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
"""Test for the nlp SingleDatasetPerformance check"""

import numpy as np
from hamcrest import assert_that, close_to, equal_to

from deepchecks.core.condition import ConditionCategory
from deepchecks.nlp.checks.model_evaluation import ConfusionMatrixReport
from deepchecks.utils.strings import format_number, format_percent
from tests.base.utils import equal_condition_result


def test_defaults(text_classification_dataset_mock):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(text_classification_dataset_mock,
                       predictions=['0', '1', '1'])

    confusion_matrix = result.value.to_numpy()

    # Assert
    assert_that(list(text_classification_dataset_mock.label), equal_to(['0', '0', '1']))
    assert_that(confusion_matrix[0][0], close_to(1, 0.001))
    assert_that(confusion_matrix.shape[0], close_to(2, 0.001))


def test_run_default_scorer_string_class(text_classification_string_class_dataset_mock):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'])

    confusion_matrix = result.value.to_numpy()

    # Assert
    assert_that(list(text_classification_string_class_dataset_mock.label), equal_to(['wise', 'meh', 'meh']))
    assert_that(confusion_matrix[0][0], close_to(1, 0.001))
    assert_that(confusion_matrix.shape[0], close_to(2, 0.001))


def test_run_default_scorer_string_class_new_cats_in_model_classes(text_classification_string_class_dataset_mock):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'new', 'meh'])

    confusion_matrix = result.value.to_numpy()

    # Assert
    assert_that(list(text_classification_string_class_dataset_mock.label), equal_to(['wise', 'meh', 'meh']))
    assert_that(confusion_matrix[0][0], close_to(1, 0.001))
    assert_that(confusion_matrix.shape[0], close_to(3, 0.001))


def test_run_tweet_emotion(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(tweet_emotion_train_test_textdata[0],
                       predictions=tweet_emotion_train_test_predictions[0])

    confusion_matrix = result.value.to_numpy()

    # Assert
    assert_that(confusion_matrix[0][0], close_to(1160, 0.001))
    assert_that(confusion_matrix.shape[0], close_to(4, 0.001))


def test_condition_misclassified_samples_lower_than_raises_error(tweet_emotion_train_test_textdata,
                                                                 tweet_emotion_train_test_predictions):

    # Arrange
    _, test_ds = tweet_emotion_train_test_textdata
    _, test_preds = tweet_emotion_train_test_predictions

    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=-0.1) \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=1.1)

    # Act
    result = check.run(test_ds, predictions=test_preds)

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


def test_condition_misclassified_samples_lower_than_passes(tweet_emotion_train_test_textdata,
                                                           tweet_emotion_train_test_predictions):

    # Arrange
    _, test_ds = tweet_emotion_train_test_textdata
    _, test_preds = tweet_emotion_train_test_predictions

    threshold = 0.1

    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=threshold)

    # Act
    result = check.run(test_ds, predictions=test_preds)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details = 'All misclassified confusion matrix cells contain less than ' \
                  f'{format_percent(threshold)} of the data.'
    ))


def test_condition_misclassified_samples_lower_than_fails(tweet_emotion_train_test_textdata,
                                                          tweet_emotion_train_test_predictions):

    # Arrange
    _, test_ds = tweet_emotion_train_test_textdata
    _, test_preds = tweet_emotion_train_test_predictions

    threshold = 0.01
    thresh_samples = round(np.ceil(threshold * len(test_ds)))

    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=threshold)

    # Act
    result = check.run(test_ds, predictions=test_preds)

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

    x, y = max_misclassified_cell_idx
    max_misclassified_samples = confusion_matrix[x][y]
    max_misclassified_samples_ratio = max_misclassified_samples / len(test_ds)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details = f'Detected {n_cells_above_thresh} misclassified confusion matrix cell(s) each one ' \
                  f'containing more than {format_percent(threshold)} of the data. ' \
                  f'Largest misclassified cell ({format_percent(max_misclassified_samples_ratio)} of the data) ' \
                  f'is samples with a true value of "{class_names[x]}" and a predicted value of "{class_names[y]}".'
    ))
