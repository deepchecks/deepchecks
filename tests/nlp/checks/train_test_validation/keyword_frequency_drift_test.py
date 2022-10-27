# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module for testing the keyword frequency drift check"""

from deepchecks.nlp.checks import KeywordFrequencyDrift
from hamcrest import assert_that, close_to, contains_exactly, equal_to
from tests.base.utils import equal_condition_result


def test_with_defaults_no_drift(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    assert_that(result.value['drift_score'], close_to(0.326, 0.001))


def test_with_keywords(movie_reviews_data_positive, movie_reviews_data_negative):
    keywords_list = ['good', 'bad', 'cartoon', 'recommend']
    result = KeywordFrequencyDrift(top_n_method=keywords_list).run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.value['drift_score'], close_to(0.438, 0.001))
    assert_that(result.display[1].data[0].x, contains_exactly('bad', 'cartoon', 'good', 'recommend'))


def test_top_freqs(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift(top_n_method='top_freq')\
        .run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.value['drift_score'], close_to(0.438, 0.001))


def test_drift_score_condition(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift()\
        .add_condition_drift_score_less_than(0.3)\
        .add_condition_drift_score_less_than(0.8)\
        .run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details='The drift score 0.44 is not less than the threshold 0.3',
        name='Drift Score is Less Than 0.3'))
    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=True,
        details='The drift score 0.44 is less than the threshold 0.8',
        name='Drift Score is Less Than 0.8'))


def test_top_n_diff_condition(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift(top_n_to_show=5, drift_method='PSI')\
        .add_condition_top_n_differences_less_than(0.2)\
        .add_condition_top_n_differences_less_than(1.1)\
        .run(movie_reviews_data_positive, movie_reviews_data_negative)

    expected_keywords = ['hype', 'regret', 'sorry', 'cusack', 'hackman']
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details=f'Failed for the keywords: {expected_keywords}',
        name='Differences between the frequencies of the top N keywords are less than 0.2'))
    assert_that(result.conditions_results[1], equal_condition_result(
        is_pass=True,
        details='Passed for all of the top N keywords',
        name='Differences between the frequencies of the top N keywords are less than 1.1'))
