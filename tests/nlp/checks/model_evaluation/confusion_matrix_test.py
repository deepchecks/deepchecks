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

from hamcrest import assert_that, close_to, equal_to

from deepchecks.nlp.checks.model_evaluation import ConfusionMatrixReport


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
