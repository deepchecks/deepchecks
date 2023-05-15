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
"""Test for the NLP PredictionDrift check"""
import numpy as np
import pytest
from hamcrest import assert_that, close_to, equal_to, has_items, has_length

from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift
from tests.base.utils import equal_condition_result


def test_tweet_emotion(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions):
    # Arrange
    train, test = tweet_emotion_train_test_textdata
    train_preds, test_preds = tweet_emotion_train_test_predictions
    check = PredictionDrift().add_condition_drift_score_less_than(0.01)
    # Act
    result = check.run(train, test, train_predictions=train_preds,
                       test_predictions=test_preds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details="Found model prediction Cramer's V drift score of 0.04",
                               name='Prediction drift score < 0.01')
    ))

    assert_that(result.value['Drift score'], close_to(0.04, 0.01))


def test_tweet_emotion_no_drift(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions):
    # Arrange
    train, _ = tweet_emotion_train_test_textdata
    train_preds, _ = tweet_emotion_train_test_predictions
    check = PredictionDrift().add_condition_drift_score_less_than()
    # Act
    result = check.run(train, train, train_predictions=train_preds, test_predictions=train_preds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details="Found model prediction Cramer's V drift score of 0",
                               name='Prediction drift score < 0.15')
    ))

    assert_that(result.value['Drift score'], equal_to(0))


def test_tweet_emotion_no_drift_no_label(tweet_emotion_train_test_textdata, tweet_emotion_train_test_predictions):
    # Arrange
    train, _ = tweet_emotion_train_test_textdata
    train = TextData(train.text, task_type='text_classification', metadata=train.metadata,
                     properties=train.properties)
    train_preds, _ = tweet_emotion_train_test_predictions
    check = PredictionDrift().add_condition_drift_score_less_than()
    # Act
    result = check.run(train, train, train_predictions=train_preds, test_predictions=train_preds)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details="Found model prediction Cramer's V drift score of 0",
                               name='Prediction drift score < 0.15')
    ))

    assert_that(result.value['Drift score'], equal_to(0))


def test_just_dance_small_drift(just_dance_train_test_textdata_sampled):
    # Arrange
    train, test = just_dance_train_test_textdata_sampled
    check = PredictionDrift().add_condition_drift_score_less_than(0.1)

    # Act
    result = check.run(train, test, train_predictions=np.asarray(train.label),
                       test_predictions=np.asarray(test.label))
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details="Found model prediction Cramer's V drift score of 0.05",
                               name='Prediction drift score < 0.1')
    ))

    assert_that(result.value['Drift score'], close_to(0.05, 0.01))


def test_token_classification(small_wikiann_train_test_text_data):
    # Arrange
    train, test = small_wikiann_train_test_text_data
    check = PredictionDrift()

    # Act
    result = check.run(train, test, train_predictions=np.asarray(train.label),
                       test_predictions=np.asarray(test.label))

    # Assert
    assert_that(result.value['Drift score'], close_to(0, 0.01))


def test_token_classification_with_nones(small_wikiann_train_test_text_data):
    # Arrange
    train, test = small_wikiann_train_test_text_data
    train_label_with_nones = train.label
    train_label_with_nones[0][0] = None
    train = TextData(train.text, tokenized_text=train.tokenized_text,
                     task_type='token_classification')
    check = PredictionDrift()

    # Act
    result = check.run(train, test, train_predictions=np.asarray(train_label_with_nones),
                       test_predictions=np.asarray(test.label))

    # Assert
    assert_that(result.value['Drift score'], close_to(0, 0.01))


def test_drift_mode_proba_warnings(small_wikiann_train_test_text_data):
    # Arrange
    train, test = small_wikiann_train_test_text_data
    check = PredictionDrift(drift_mode='proba')

    # Act
    with pytest.warns(UserWarning,
                      match='Cannot use drift_mode="proba" for multi-label text classification tasks or token '
                            'classification tasks. Using drift_mode="prediction" instead.'):
        check.run(train, test, train_predictions=np.asarray(train.label), test_predictions=np.asarray(test.label))

    check = PredictionDrift()

    with pytest.warns(None) as record:
        check.run(train, test, train_predictions=np.asarray(train.label), test_predictions=np.asarray(test.label))

    assert_that(record, has_length(0))

