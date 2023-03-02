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
"""Test for the NLP PropertyLabelCorrelation check"""

from hamcrest import assert_that, close_to, equal_to, has_items

from deepchecks.nlp.checks import PropertyLabelCorrelation
from deepchecks.nlp.datasets.classification import tweet_emotion
from tests.base.utils import equal_condition_result


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_probas = tweet_emotion.load_precalculated_predictions(pred_format='probabilities')[test.index]
    check = PropertyLabelCorrelation().add_condition_feature_pps_less_than(0.1)
    # Act
    result = check.run(test, probabilities=test_probas)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=False,
                               details="Found 1 out of 6 features with PPS above threshold: {'sentiment': '0.11'}",
                               name="Properties' Predictive Power Score is less than 0.1")
    ))

    assert_that(result.value['sentiment'], close_to(0.11, 0.01))
    assert_that(result.value['text_length'], close_to(0.02, 0.01))
