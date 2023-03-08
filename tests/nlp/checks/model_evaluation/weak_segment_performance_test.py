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
"""Test for the NLP WeakSegmentsPerformance check"""
import pytest
from hamcrest import assert_that, close_to, equal_to, has_items

from deepchecks.nlp.checks import MetadataSegmentsPerformance, PropertySegmentsPerformance
from tests.base.utils import equal_condition_result


def test_tweet_emotion(tweet_emotion_train_test_textdata, tweet_emotion_train_test_probabilities):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    _, test_probas = tweet_emotion_train_test_probabilities
    check = MetadataSegmentsPerformance().add_condition_segments_relative_performance_greater_than()
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


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata, tweet_emotion_train_test_probabilities):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    _, test_probas = tweet_emotion_train_test_probabilities
    check = PropertySegmentsPerformance().add_condition_segments_relative_performance_greater_than(max_ratio_change=0.3)
    # Act
    result = check.run(test, probabilities=test_probas)
    condition_result = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, has_items(
        equal_condition_result(is_pass=True,
                               details='Found a segment with Accuracy score of 0.548 in comparison to an average score of 0.708 in sampled data.',
                               name='The relative performance of weakest segment is greater than 70% of average model performance.')
    ))

    assert_that(result.value['avg_score'], close_to(0.708, 0.001))
    assert_that(len(result.value['weak_segments_list']), equal_to(12))
    assert_that(result.value['weak_segments_list'].iloc[0, 0], close_to(0.548, 0.01))


def test_warning_of_n_top_columns(tweet_emotion_train_test_textdata, tweet_emotion_train_test_probabilities):
    _, test = tweet_emotion_train_test_textdata
    _, test_probas = tweet_emotion_train_test_probabilities
    property_check = PropertySegmentsPerformance(n_top_properties=3)
    metadata_check = MetadataSegmentsPerformance(n_top_columns=2)

    property_warning = 'Parameter n_top_properties is set to 3 to avoid long computation time. This means that the ' \
                       'check will run on the first 3 properties. If you want to run on all properties, set ' \
                       'n_top_properties to None. Alternatively, you can set parameter properties to a list of the ' \
                       'specific properties you want to run on.'

    metadata_warning = 'Parameter n_top_columns is set to 2 to avoid long computation time. This means that the check ' \
                       'will run on the first 2 metadata columns. If you want to run on all metadata columns, set ' \
                       'n_top_columns to None. Alternatively, you can set parameter columns to a list of the specific ' \
                       'metadata columns you want to run on.'

    # Assert
    with pytest.warns(UserWarning, match=property_warning):
        _ = property_check.run(test, probabilities=test_probas)
    with pytest.warns(UserWarning, match=metadata_warning):
        _ = metadata_check.run(test, probabilities=test_probas)
