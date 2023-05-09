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

from sys import platform
from hamcrest import assert_that, close_to

from deepchecks.nlp.checks import TextEmbeddingsDrift
from tests.base.utils import equal_condition_result


def test_tweet_emotion_no_drift(tweet_emotion_train_test_textdata_sampled):
    # Arrange
    train, _ = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift(with_display=False)
    # Act
    result = check.run(train, train)

    assert_that(result.value['domain_classifier_drift_score'], close_to(0, 0.01))


def test_tweet_emotion(tweet_emotion_train_test_textdata_sampled):
    # Arrange
    train, test = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift()
    # Act
    result = check.run(train, test)

    # UMAP uses numba, which uses different random seeds on different OSes. And that can't be changed ATM.
    # For more, see https://github.com/lmcinnes/umap/issues/183
    if platform.startswith('win'):
        assert_that(result.value['domain_classifier_drift_score'], close_to(0.11, 0.01))
    else:
        assert_that(result.value['domain_classifier_drift_score'], close_to(0.24, 0.01))


def test_reduction_method(tweet_emotion_train_test_textdata_sampled):
    # Arrange
    train, test = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift(dimension_reduction_method='PCA')
    # Act
    result = check.run(train, test)

    assert_that(result.value['domain_classifier_drift_score'], close_to(0.17, 0.01))

    # Make sure uses PCA with auto + with_display false:
    check = TextEmbeddingsDrift(dimension_reduction_method='auto')
    # Act
    result = check.run(train, test, with_display=False)

    assert_that(result.value['domain_classifier_drift_score'], close_to(0.17, 0.01))

    # Make sure doesn't use embeddings if none:
    check = TextEmbeddingsDrift(dimension_reduction_method='none')
    # Act
    result = check.run(train, test)

    assert_that(result.value['domain_classifier_drift_score'], close_to(0.14, 0.01))


def test_max_drift_score_condition_pass(tweet_emotion_train_test_textdata_sampled):
    # Arrange
    train, test = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift().add_condition_overall_drift_value_less_than()

    # Act
    result = check.run(train, test, with_display=False)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=True,
        details='Found drift value of: 0.17, corresponding to a domain classifier AUC of: 0.58',
        name='Drift value is less than 0.25',
    ))


def test_max_drift_score_condition_fail(tweet_emotion_train_test_textdata_sampled):
    # Arrange
    train, test = tweet_emotion_train_test_textdata_sampled
    check = TextEmbeddingsDrift().add_condition_overall_drift_value_less_than(0.15)

    # Act
    result = check.run(train, test, with_display=False)
    condition_result, *_ = check.conditions_decision(result)

    # Assert
    assert_that(condition_result, equal_condition_result(
        is_pass=False,
        name='Drift value is less than 0.15',
        details='Found drift value of: 0.17, corresponding to a domain classifier AUC of: 0.58'
    ))
