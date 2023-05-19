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
"""Test for the properties module"""
import os
import pathlib
import timeit
from unittest.mock import patch

import numpy as np
import pytest
from hamcrest import *

from deepchecks.nlp.utils.text_properties import MODELS_STORAGE, calculate_builtin_properties, get_transformer_model


def mock_fn(*args, **kwargs):  # pylint: disable=unused-argument
    return [0] * 20_000


@pytest.fixture(name='manual_text_data_for_properties')
def text_data_fixture():
    """Mock data for a calculating text properties."""
    text_data = {
        'url_data': [
            'Please contact me at abc.ex@example.com.',
            'For more information, visit our website: https://deepchecks.com/.',
            'Email us at info@example.com or visit our website http://www.example.com for assistance.',
            'For any inquiries, send an email to support@example.com.',
            'The results were found at http://www.google.com and it redirects to'
            'https://www.deepchecks.com and there we can find the links to all social medias such'
            'as http://gmail.com, https://fb.com, https://www.deepchecks.com, https://www.google.com'
        ],
        'email_data': [
            'Please send your inquiries to info@example.com or support@example.com. We are happy to assist you.',
            'Contact us at john.doe@example.com or jane.smith@example.com for further information\
            Looking forward to hearing from you.',
            'For any questions or concerns, email sales@example.com. We are here to help.',
            'Hello, this is a text without email address@asx',
            'You can contact me directly at samantha.wilson@example.com or use the\
            team email address marketing@example.com.',
            'If you have any feedback or suggestions, feel free to email us at feedback@example.com,\
            support@example.com, feedback@example.com.'
        ]
    }
    return text_data


@patch('deepchecks.nlp.utils.text_properties.run_available_kwargs', mock_fn)
def test_that_warning_is_shown_for_big_datasets():
    # Arrange
    raw_text = ['This is a test sentence.'] * 20_000

    match_text = r'Calculating the properties \[\'Toxicity\'\] on a large dataset may take a long time.' \
                 r' Consider using a smaller sample size or running this code on better hardware. Consider using a ' \
                 r'GPU or a similar device to run these properties.'

    # Act
    with pytest.warns(UserWarning,
                      match=match_text):
        result = calculate_builtin_properties(raw_text, include_properties=['Toxicity'],
                                              include_long_calculation_properties=True)[0]

    # Assert
    assert len(result['Toxicity']) == len(raw_text)


def test_calculate_lexical_density_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Lexical Density'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Lexical Density'])[0]

    # Assert
    assert_that(result['Lexical Density'][0: 10], equal_to([94.44, 93.75, 100.0, 91.67,
                                                            87.5, 100.0, 100.0, 100.0, 91.67, 91.67]))
    assert_that(result_none_text['Lexical Density'], equal_to([np.nan]))


def test_calculate_unique_noun_count_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Unique Noun Count'],
                                          include_long_calculation_properties=True)[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Unique Noun Count'],
                                                    include_long_calculation_properties=True)[0]

    # Assert
    assert_that(result['Unique Noun Count'][0: 10], equal_to([9, 2, 3, 3, 4, 10, np.nan, 2, 7, 5]))
    assert_that(result_none_text['Unique Noun Count'], equal_to([np.nan]))


def test_calculate_average_sentence_length_property(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Average Sentence Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Average Sentence Length'])[0]

    # Assert
    assert_that(result['Average Sentence Length'][0: 10], equal_to([6, 7, 11, 12, 8, 19, 3, 9, 12, 7]))
    assert_that(result_none_text['Average Sentence Length'], equal_to([np.nan]))


def test_calculate_readability_score_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Readability Score'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Readability Score'])[0]

    # Assert
    assert_that(result['Readability Score'][0: 10], equal_to([102.045, 97.001, 80.306, 67.755, 77.103,
                                                            71.782, np.nan, 75.5, 70.102, 95.564]))
    assert_that(result_none_text['Readability Score'], equal_to([np.nan]))


def test_calculate_count_unique_urls(manual_text_data_for_properties):

    # Arrange
    text_data = manual_text_data_for_properties['url_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Count Unique URLs'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Count Unique URLs'])[0]

    # Assert
    assert_that(result['Count Unique URLs'], equal_to([0, 1, 1, 0, 5]))
    assert_that(result_none_text['Count Unique URLs'], equal_to([0]))


def test_calculate_count_urls(manual_text_data_for_properties):

    # Arrange
    text_data = manual_text_data_for_properties['url_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Count URLs'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Count URLs'])[0]

    # Assert
    assert_that(result['Count URLs'], equal_to([0, 1, 1, 0, 6]))
    assert_that(result_none_text['Count URLs'], equal_to([0]))


def test_calculate_count_unique_email_addresses(manual_text_data_for_properties):

    # Arrange
    text_data = manual_text_data_for_properties['email_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Count Unique Email Address'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Count Unique Email Address'])[0]

    # Assert
    assert_that(result['Count Unique Email Address'], equal_to([2, 2, 1, 0, 2, 2]))
    assert_that(result_none_text['Count Unique Email Address'], equal_to([0]))


def test_calculate_count_email_addresses(manual_text_data_for_properties):

    # Arrange
    text_data = manual_text_data_for_properties['email_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Count Email Address'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Count Email Address'])[0]

    # Assert
    assert_that(result['Count Email Address'], equal_to([2, 2, 1, 0, 2, 3]))
    assert_that(result_none_text['Count Email Address'], equal_to([0]))


def test_calculate_count_unique_syllables(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Count Unique Syllables'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Count Unique Syllables'])[0]

    # Assert
    assert_that(result['Count Unique Syllables'][0: 10], equal_to([15, 11, 9, 21, 13, 17, np.nan, 8, 20, 18]))
    assert_that(result_none_text['Count Unique Syllables'], equal_to([np.nan]))


def test_calculate_reading_time(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Reading Time'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Reading Time'])[0]

    # Assert
    assert_that(result['Reading Time'][0: 10], equal_to([1.26, 1.25, 0.81, 1.35, 1.44,
                                                         1.88, 0.48, 0.71, 1.53, 1.56]))
    assert_that(result_none_text['Reading Time'], equal_to([0.00]))


def test_calculate_sentence_length(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Sentence Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Sentence Length'])[0]

    # Assert
    assert_that(result['Sentence Length'][0: 10], equal_to([3, 2, 1, 2, 2, 1, np.nan, 1, 2, 3]))
    assert_that(result_none_text['Sentence Length'], equal_to([np.nan]))


def test_calculate_average_syllable_count(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Average Syllable Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Average Syllable Length'])[0]

    # Assert
    assert_that(result['Average Syllable Length'][0: 10], equal_to([7.0, 8.5, 15.0, 18.0, 11.5,
                                                                    26.0, np.nan, 13.0, 17.0, 9.0]))
    assert_that(result_none_text['Average Syllable Length'], equal_to([np.nan]))


def test_ignore_properties():

    # Arrange
    test_text = ['This is simple sentence.']
    expected_properties = ['Text Length', 'Average Word Length', 'Max Word Length',
                           '% Special Characters', 'Language','Sentiment', 'Subjectivity',
                           'Lexical Density', 'Readability Score', 'Average Sentence Length']
    # Act
    result = calculate_builtin_properties(test_text, ignore_properties=['Unique Noun Count',
                                                                        'Toxicity', 'Fluency',
                                                                        'Formality'])[0]
    # Assert
    for prop in result:
        assert_that(expected_properties, has_item(prop))


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_properties_models_download():
    # Arrange
    model_name = 'unitary/toxic-bert'
    model_path = MODELS_STORAGE / f"models--{model_name.replace('/', '--')}"
    onnx_model_path = MODELS_STORAGE / 'onnx' / model_name
    quantized_model_path = MODELS_STORAGE / 'onnx' / 'quantized' / model_name

    # Act
    model_download_time = timeit.timeit(
        stmt='fn()',
        number=1,
        globals={'fn': lambda: get_transformer_model(
            property_name='',
            model_name=model_name
        )}
    )

    # Assert
    assert MODELS_STORAGE.exists() and MODELS_STORAGE.is_dir()
    assert model_path.exists() and model_path.is_dir()
    assert onnx_model_path.exists() and onnx_model_path.is_dir()

    # Act
    get_transformer_model(property_name='', model_name=model_name, quantize_model=True)

    # Assert
    assert quantized_model_path.exists() and quantized_model_path.is_dir()

    # Act
    model_creation_time = timeit.timeit(
        stmt='fn()',
        number=1,
        globals={'fn': lambda: get_transformer_model(
            property_name='',
            model_name=model_name,
            quantize_model=True
        )}
    )

    # Assert
    assert model_creation_time <= model_download_time * 0.1


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_properties_models_download_into_provided_directory():
    directory = pathlib.Path(__file__).absolute().parent / '.models'
    model_name = 'unitary/toxic-bert'
    model_path = MODELS_STORAGE / f"models--{model_name.replace('/', '--')}"
    onnx_model_path = MODELS_STORAGE / 'onnx' / model_name

    # Act
    get_transformer_model(property_name='', model_name=model_name, models_storage=directory)

    # Assert
    assert MODELS_STORAGE.exists() and MODELS_STORAGE.is_dir()
    assert model_path.exists() and model_path.is_dir()
    assert onnx_model_path.exists() and onnx_model_path.is_dir()


def test_english_only_properties_calculation_with_not_english_samples():
    # Arrange
    text = [
        'Explicit is better than implicit',
        'Сьогодні чудова погода',
        'London is the capital of Great Britain'
    ]
    # Act
    properties, properties_types = calculate_builtin_properties(
        raw_text=text,
        include_properties=['Sentiment', 'Language', 'Text Length']
    )
    # Assert
    assert_that(properties, has_entries({
        'Sentiment': contains_exactly(close_to(0.5, 0.01), same_instance(np.nan), close_to(0.8, 0.01)),
        'Language': contains_exactly('en', 'uk', 'en'),
        'Text Length': contains_exactly(*[len(it) for it in text]),
    }))  # type: ignore
    assert_that(properties_types, has_entries({
        'Sentiment': 'numeric',
        'Language': 'categorical',
        'Text Length': 'numeric',
    }))  # type: ignore


