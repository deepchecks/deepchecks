from deepchecks.nlp.checks import KeywordFrequencyDrift


def test_with_defaults_no_drift(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    print(result.value)
    result.save_as_html('/Users/solyarkoni/Documents/results/1')


def test_with_keywords(movie_reviews_data_positive, movie_reviews_data_negative):
    keywords_list = ['good', 'fun', 'worse', 'boring', 'action']
    result = KeywordFrequencyDrift(top_n_method=keywords_list).run(movie_reviews_data_positive, movie_reviews_data_negative)
    print(result.value)
    result.save_as_html('/Users/solyarkoni/Documents/results/2')