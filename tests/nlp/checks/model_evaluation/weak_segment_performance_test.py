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
"""Test for the nlp WeakSegmentsPerformance check"""

from hamcrest import assert_that, close_to, equal_to, has_items

from deepchecks.nlp.checks import AdditionalDataSegmentsPerformance, PropertySegmentsPerformance
from deepchecks.nlp.datasets.classification import tweet_emotion
from tests.base.utils import equal_condition_result


def test_tweet_emotion(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities')[test.index]
    check = AdditionalDataSegmentsPerformance().add_condition_segments_relative_performance_greater_than()
    # Act
    result = check.run(test, probabilities=test_probas)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found a segment with Accuracy score of 0.305 in comparison to an average score of 0.708 in sampled data.',
                               name='The relative performance of weakest segment is greater than 80% of average model performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.708, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(6))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.305, 0.01))


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities')[test.index]
    check = PropertySegmentsPerformance().add_condition_segments_relative_performance_greater_than()
    # Act
    result = check.run(test, probabilities=test_probas)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details='Found a segment with Accuracy score of 0.305 in comparison to an average score of 0.708 in sampled data.',
                               name='The relative performance of weakest segment is greater than 80% of average model performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.708, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(6))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.305, 0.01))
