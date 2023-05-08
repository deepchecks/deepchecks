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
"""Test for the default suites"""
from deepchecks.nlp.suites import model_evaluation, full_suite
from tests.common import get_expected_results_length, validate_suite_result


def test_full_suite(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions,
                    tweet_emotion_train_test_probabilities):
    # Arrange
    train_data, test_data = tweet_emotion_train_test_textdata
    train_preds, test_preds = tweet_emotion_train_test_predictions
    train_probas, test_probas = tweet_emotion_train_test_probabilities

    kwargs = dict(train_dataset=train_data, test_dataset=test_data, train_predictions=train_preds,
                  test_predictions=test_preds, train_probabilities=train_probas, test_probabilities=test_probas)

    # Act
    suite = full_suite(imaginary_kwarg='just to make sure all checks have kwargs in the init')
    result = suite.run(**kwargs)

    # Assert
    length = get_expected_results_length(suite, kwargs)
    validate_suite_result(result, length)


def test_model_eval_suite_with_model_classes_argument(tweet_emotion_train_test_textdata,
                                                      tweet_emotion_train_test_predictions,
                                                      tweet_emotion_train_test_probabilities):
    # Arrange
    train_data, test_data = tweet_emotion_train_test_textdata
    train_preds, test_preds = tweet_emotion_train_test_predictions
    train_probas, test_probas = tweet_emotion_train_test_probabilities

    kwargs = dict(train_dataset=train_data, test_dataset=test_data, train_predictions=train_preds,
                  test_predictions=test_preds, train_probabilities=train_probas, test_probabilities=test_probas,
                  model_classes=['anger', 'happiness', 'optimism', 'sadness'])

    # Act
    suite = model_evaluation()
    result = suite.run(**kwargs)

    # Assert
    length = get_expected_results_length(suite, kwargs)
    validate_suite_result(result, length)
