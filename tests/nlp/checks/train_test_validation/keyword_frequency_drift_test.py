from deepchecks.nlp.checks import KeywordFrequencyDrift
from hamcrest import assert_that, close_to, contains_exactly

def test_with_defaults_no_drift(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    assert_that(result.value, close_to(0.529, 0.05))
    result.save_as_html('/Users/solyarkoni/Documents/results/1')


def test_with_keywords(movie_reviews_data_positive, movie_reviews_data_negative):
    keywords_list = ['good', 'fun', 'worse', 'boring', 'action']
    result = KeywordFrequencyDrift(top_n_method=keywords_list).run(movie_reviews_data_positive, movie_reviews_data_negative)
    assert_that(result.value, close_to(0.579, 0.05))
    assert_that(result.display[0].data[0].x.tolist(), contains_exactly('action', 'boring', 'fun', 'good', 'worse'))