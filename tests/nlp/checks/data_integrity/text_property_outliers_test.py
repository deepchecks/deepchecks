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
"""Test for the NLP TextPropertyOutliers check"""

from hamcrest import assert_that, close_to, equal_to, raises, calling

from deepchecks.core.errors import NotEnoughSamplesError
from deepchecks.nlp.checks import TextPropertyOutliers


def test_tweet_emotion_properties(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = TextPropertyOutliers()
    # Act
    result = check.run(test)

    # Assert
    assert_that(len(result.value['sentiment']['indices']), equal_to(137))
    assert_that(result.value['sentiment']['lower_limit'], close_to(-0.72, 0.01))
    assert_that(result.value['sentiment']['upper_limit'], close_to(0.74, 0.01))

    assert_that(len(result.value['text_length']['indices']), equal_to(0))
    assert_that(result.value['text_length']['lower_limit'], equal_to(6))
    assert_that(result.value['text_length']['upper_limit'], equal_to(160))

    # Categorical property:
    assert_that(len(result.value['language']['indices']), equal_to(55))
    assert_that(result.value['language']['lower_limit'], close_to(0.007, 0.001))
    assert_that(result.value['language']['upper_limit'], equal_to(None))

    # Assert display
    assert_that(len(result.display), equal_to(6))
    assert_that(result.display[4], equal_to('<h5><b>Properties With No Outliers Found</h5></b>'))


def test_not_enough_samples(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    check = TextPropertyOutliers(min_samples=6000)

    assert_that(calling(check.run).with_args(test),
                raises(NotEnoughSamplesError, 'Need at least 6000 non-null samples to calculate outliers.'))

