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
"""Test for the NLP TextEmbeddingsDrift check"""

from hamcrest import assert_that, close_to, has_items

from deepchecks.nlp.checks import TextEmbeddingsDrift
from tests.base.utils import equal_condition_result


def test_tweet_emotion(tweet_emotion_train_test_textdata):
    # Arrange
    train, test = tweet_emotion_train_test_textdata
    check = TextEmbeddingsDrift()
    # Act
    result = check.run(train, test)


    assert_that(result.value['Drift score'], close_to(0.23, 0.01))


# def test_tweet_emotion_no_drift(tweet_emotion_train_test_textdata):
#     # Arrange
#     train, _ = tweet_emotion_train_test_textdata
#     check = TextEmbeddingsDrift().add_condition_drift_score_less_than()
#     # Act
#     result = check.run(train, train)
#     condition_result = check.conditions_decision(result)
#
#     # Assert
#     assert_that(condition_result, has_items(
#         equal_condition_result(is_pass=True,
#                                details="Label's drift score Cramer's V is 0",
#                                name='categorical drift score < 0.15 and numerical drift score < 0.15 for label drift')
#     ))
#
#     assert_that(result.value['Drift score'], close_to(0, 0.01))
