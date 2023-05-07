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

    # Assert
    assert_that(list(text_classification_dataset_mock.label), equal_to(['0', '0', '1']))
    assert_that(result.value[0][0], close_to(1, 0.001))
    assert_that(result.value.shape[0], close_to(2, 0.001))


def test_run_default_scorer_string_class(text_classification_string_class_dataset_mock):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'wise', 'meh'])

    # Assert
    assert_that(list(text_classification_string_class_dataset_mock.label), equal_to(['wise', 'meh', 'meh']))
    assert_that(result.value[0][0], close_to(1, 0.001))
    assert_that(result.value.shape[0], close_to(2, 0.001))


def test_run_default_scorer_string_class_new_cats_in_model_classes(text_classification_string_class_dataset_mock):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(text_classification_string_class_dataset_mock,
                       predictions=['wise', 'new', 'meh'])

    # Assert
    assert_that(list(text_classification_string_class_dataset_mock.label), equal_to(['wise', 'meh', 'meh']))
    assert_that(result.value[0][0], close_to(1, 0.001))
    assert_that(result.value.shape[0], close_to(3, 0.001))


def test_run_tweet_emotion(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions):
    # Arrange
    check = ConfusionMatrixReport()

    # Act
    result = check.run(tweet_emotion_train_test_textdata[0],
                       predictions=tweet_emotion_train_test_predictions[0])

    # Assert
    assert_that(result.value[0][0], close_to(1160, 0.001))
    assert_that(result.value.shape[0], close_to(4, 0.001))


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
        details='Exception in condition: ValidationError: Condition requires the parameter "misclassified_samples_threshold" '
                'to be between 0 and 1 inclusive but got -0.1',
        category=ConditionCategory.ERROR
    ))

    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_number(1.1 * 100)}% of the total samples',
        details='Exception in condition: ValidationError: Condition requires the parameter "misclassified_samples_threshold" '
                'to be between 0 and 1 inclusive but got 1.1',
        category=ConditionCategory.ERROR
    ))


def test_condition_misclassified_samples_lower_than_passes(tweet_emotion_train_test_textdata,
                                                           tweet_emotion_train_test_predictions):

    # Arrange
    _, test_ds = tweet_emotion_train_test_textdata
    _, test_preds = tweet_emotion_train_test_predictions

    threshold = 0.1
    thresh_samples = round(np.ceil(threshold * len(test_ds)))

    check = ConfusionMatrixReport() \
            .add_condition_misclassified_samples_lower_than_condition(misclassified_samples_threshold=threshold)

    # Act
    result = check.run(test_ds, predictions=test_preds)

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=True,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details='Number of samples in each of the misclassified cells in the confusion matrix is '
                f'lesser than the threshold ({thresh_samples}) based on the given misclassified_samples_threshold ratio'
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

    confusion_matrix = result.value
    m, n = confusion_matrix.shape[0], confusion_matrix.shape[1]

    n_cells_above_thresh = 0
    max_samples_in_cell_above_thresh = thresh_samples

    # Looping over the confusion matrix and checking only the misclassified cells
    for i in range(m):
        for j in range(n):
            # omitting the principal axis of the confusion matrix
            if i != j:
                n_samples = confusion_matrix[i][j]

                if n_samples > thresh_samples:
                    n_cells_above_thresh += 1

                    if n_samples > max_samples_in_cell_above_thresh:
                        max_samples_in_cell_above_thresh = n_samples

    # Assert
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        name=f'Misclassified cell size lower than {format_percent(threshold)} of the total samples',
        details = f'The confusion matrix has {n_cells_above_thresh} cells with samples greater than the '
                   f'threshold ({thresh_samples}) based on the given misclassified_samples_threshold ratio. '
                   f'The worst performing cell has {max_samples_in_cell_above_thresh} samples'
    ))
