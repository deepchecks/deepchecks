from deepchecks.nlp.checks import KeywordFrequencyDrift
from hamcrest import assert_that, close_to, contains_exactly
from tests.base.utils import equal_condition_result


def test_with_defaults_no_drift(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    assert_that(result.value['drift_score'], close_to(0.301, 0.05))
    # result.save_as_html('/Users/solyarkoni/Documents/results/1.html')
    # print(result.value['drift_score'])


def test_with_keywords(movie_reviews_data_positive, movie_reviews_data_negative):
    keywords_list = ['blech', 'great', 'amaz', 'good', 'bad']
    result = KeywordFrequencyDrift(top_n_method=keywords_list).run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.value['drift_score'], close_to(0.399, 0.05))
    assert_that(result.display[0].data[0].x.tolist(), contains_exactly('amaz', 'bad', 'blech', 'good'))
    # result.save_as_html('/Users/solyarkoni/Documents/results/2.html')
    # print(result.value['drift_score'])


def test_drift_score_condition_fail(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift().add_condition_drift_score_less_than(0.3)\
        .run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details='The drift score 0.4 is not less than the threshold 0.3',
        name='Drift Score is Less Than 0.3'))


def test_top_n_diff_condition_fail(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift(top_n_to_show=5, drift_method='PSI').add_condition_top_n_differences_less_than(0.2)\
        .run(movie_reviews_data_positive, movie_reviews_data_negative)

    expected_keywords = ['awwwww', 'wafflemovy', 'erotic', 'bah', 'rabl']
    assert_that(result.conditions_results[0], equal_condition_result(
        is_pass=False,
        details=f'Failed for the keywords: {expected_keywords}',
        name='Diffrences between the frequencies of the top N keywords are less than 0.2'))