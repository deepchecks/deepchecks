from deepchecks.nlp.checks import KeywordFrequencyDrift


def test_with_defaults_no_drift(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    print(result.value)


def test_with_defaults_with_drift(movie_reviews_data_positive, movie_reviews_data_negative):
    result = KeywordFrequencyDrift().run(movie_reviews_data_positive, movie_reviews_data_negative)
    print(result.value)
