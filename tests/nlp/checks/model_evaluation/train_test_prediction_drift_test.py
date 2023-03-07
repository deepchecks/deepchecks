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
"""Test for the nlp PredictionDrift check"""

from hamcrest import assert_that, close_to, has_items

from deepchecks.nlp.checks import TrainTestPredictionDrift
from deepchecks.nlp.datasets.classification import tweet_emotion
from tests.base.utils import equal_condition_result


def test_tweet_emotion(tweet_emotion_train_test_textdata):
    # Arrange
    train, test = tweet_emotion_train_test_textdata
    preds = tweet_emotion.load_precalculated_predictions()
    check = TrainTestPredictionDrift().add_condition_drift_score_less_than(0.01)
    # Act
    result = check.run(train, test, train_predictions=list(preds[train.index]),
                       test_predictions=list(preds[test.index]))
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details="Found model prediction Cramer's V drift score of 0.04",
                               name='categorical drift score < 0.01 and numerical drift score < 0.15')
    ))

    assert_that(result.value['Drift score'], close_to(0.04, 0.01))

# TODO: Bug in combination of sampling mechanism and dummy model
# def test_tweet_emotion_no_drift(tweet_emotion_train_test_textdata):
#     # Arrange
#     train, _ = tweet_emotion_train_test_textdata
#     preds = list(tweet_emotion.load_precalculated_predictions()[train.index])
#     check = TrainTestPredictionDrift().add_condition_drift_score_less_than(0.1)
#     # Act
#     result = check.run(train, train, train_predictions=preds, test_predictions=preds)
#     condition_result = check.conditions_decision(result)
#
#     # Assert
#     assert_that(condition_result, has_items(
#         equal_condition_result(is_pass=True,
#                                details="Found model prediction Cramer's V drift score of 0.04",
#                                name='categorical drift score < 0.1 and numerical drift score < 0.075')
#     ))
#
#     assert_that(result.value['Drift score'], close_to(0.04, 0.01))
