from deepchecks.nlp.checks import KeywordFrequencyDrift


def test_with_defaults(movie_reviews_data):
    train_data, test_data = movie_reviews_data
    result = KeywordFrequencyDrift().run(train_data, test_data)
    print(result.value)

