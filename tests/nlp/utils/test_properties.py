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

import numpy as np
import pytest
import uuid
from hamcrest import *

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.utils.text_properties import (_sample_for_property, calculate_builtin_properties,
                                                  english_text, TOXICITY_MODEL_NAME_ONNX)
from deepchecks.nlp.utils.text_properties_models import MODELS_STORAGE, _get_transformer_model_and_tokenizer


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


# TODO: Fix test (problem with pytorch versions)
# @patch('deepchecks.nlp.utils.text_properties.run_available_kwargs', mock_fn)
# def test_that_warning_is_shown_for_big_datasets():
#     # Arrange
#     raw_text = ['This is a test sentence.'] * 20_000
#
#     match_text = r'Calculating the properties \[\'Toxicity\'\] on a large dataset may take a long time.' \
#                  r' Consider using a smaller sample size or running this code on better hardware. Consider using a ' \
#                  r'GPU or a similar device to run these properties.'
#
#     # Act
#     with pytest.warns(UserWarning,
#                       match=match_text):
#         result = calculate_builtin_properties(raw_text, include_properties=['Toxicity'],
#                                               include_long_calculation_properties=True)[0]
#
#     # Assert
#     assert len(result['Toxicity']) == len(raw_text)


def test_calculate_text_length():
    # Arrange
    text = ['This is a test sentence.', 'This is another test sentence.']

    # Act
    result = calculate_builtin_properties(text, include_properties=['Text Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Text Length'])[0]
    result_empty_string = calculate_builtin_properties([''], include_properties=['Text Length'])[0]

    # Assert
    assert_that(result['Text Length'], equal_to([24, 30]))
    assert_that(result_none_text['Text Length'], equal_to([np.nan]))
    assert_that(result_empty_string['Text Length'], equal_to([0]))


def test_average_word_length():
    # Arrange
    text = ['This is a, test sentence.', 'This is another !!! test sentence.', 'וואלק זה משפט בעברית אפילו יא ווראדי']

    # Act
    result = calculate_builtin_properties(text, include_properties=['Average Word Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Average Word Length'])[0]
    result_empty_string = calculate_builtin_properties([''], include_properties=['Average Word Length'])[0]

    # Assert
    assert_that(result['Average Word Length'], equal_to([19 / 5, 25 / 5, 30 / 7]))
    assert_that(result_none_text['Average Word Length'], equal_to([np.nan]))
    assert_that(result_empty_string['Average Word Length'], equal_to([0]))


def test_max_word_length():
    # Arrange
    text = ['This is a, test sentence.', 'This is another !!! test sentence.', 'וואלק זה משפט בעברית אפילו יא ווראדי']

    # Act
    result = calculate_builtin_properties(text, include_properties=['Max Word Length'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Max Word Length'])[0]
    result_empty_string = calculate_builtin_properties([''], include_properties=['Max Word Length'])[0]

    # Assert
    assert_that(result['Max Word Length'], equal_to([8, 8, 6]))
    assert_that(result_none_text['Max Word Length'], equal_to([np.nan]))
    assert_that(result_empty_string['Max Word Length'], equal_to([0]))


def test_percentage_special_characters():
    # Arrange
    text = ['This is a, test sentence.', 'This is another <|> test sentence.', 'וואלק זה משפט בעברית אפילו יא ווראדי']

    # Act
    result = calculate_builtin_properties(text, include_properties=['% Special Characters'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['% Special Characters'])[0]
    result_empty_string = calculate_builtin_properties([''], include_properties=['% Special Characters'])[0]

    # Assert
    assert_that(result['% Special Characters'], equal_to([0, 3 / 34, 0]))
    assert_that(result_none_text['% Special Characters'], equal_to([np.nan]))
    assert_that(result_empty_string['% Special Characters'], equal_to([0]))


def test_percentage_punctuation():
    # Arrange
    text = ['This is a, test sentence.', 'This is another <|> test sentence.', 'וואלק זה משפט בעברית אפילו יא ווראדי']

    # Act
    result = calculate_builtin_properties(text, include_properties=['% Punctuation'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['% Punctuation'])[0]
    result_empty_string = calculate_builtin_properties([''], include_properties=['% Punctuation'])[0]

    # Assert
    assert_that(result['% Punctuation'], equal_to([2 / 25, 4 / 34, 0]))
    assert_that(result_none_text['% Punctuation'], equal_to([np.nan]))
    assert_that(result_empty_string['% Punctuation'], equal_to([0]))


def test_calculate_lexical_density_property(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Lexical Density'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Lexical Density'])[0]

    # Assert
    assert_that(result['Lexical Density'][0: 10], equal_to([88.24, 92.86, 100.0, 91.67,
                                                            87.5, 100.0, 100.0, 100.0, 91.3, 95.45]))
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
    result = calculate_builtin_properties(test_text, include_properties=['Average Words Per Sentence'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Average Words Per Sentence'])[0]

    # Assert
    assert_that(result['Average Words Per Sentence'][0: 10], equal_to([5.667, 7.0, 11.0, 12.0, 8.0, 19.0, 3.0, 9.0,
                                                                       11.5, 7.333]))
    assert_that(result_none_text['Average Words Per Sentence'], equal_to([np.nan]))


def test_batch_size_change(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text
    batch_sizes = [1, 8, 16, 32, 64]

    # Act
    for batch in batch_sizes:
        result = calculate_builtin_properties(test_text, include_properties=['Average Words Per Sentence',
                                                                             'Unique Noun Count'],
                                              batch_size=batch)[0]
        result_none_text = calculate_builtin_properties([None], include_properties=['Average Words Per Sentence',
                                                                                    'Unique Noun Count'],
                                                        batch_size=batch)[0]

        # Assert
        assert_that(result['Average Words Per Sentence'][0: 10], equal_to([5.667, 7.0, 11.0, 12.0, 8.0, 19.0, 3.0, 9.0,
                                                                           11.5, 7.333]))
        assert_that(result['Unique Noun Count'][0: 10], equal_to([9, 2, 3, 3, 4, 10, np.nan, 2, 7, 5]))
        assert_that(result_none_text['Average Words Per Sentence'], equal_to([np.nan]))
        assert_that(result_none_text['Unique Noun Count'], equal_to([np.nan]))


def test_calculate_readability_score_property(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Reading Ease'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Reading Ease'])[0]

    # Assert
    assert_that(result['Reading Ease'][0: 10], equal_to([96.577, 97.001, 80.306, 67.755, 77.103, 71.782,
                                                              np.nan, 75.5, 70.102, 95.564]))
    assert_that(result_none_text['Reading Ease'], equal_to([np.nan]))


def test_calculate_count_unique_urls(manual_text_data_for_properties):
    # Arrange
    text_data = manual_text_data_for_properties['url_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Unique URLs Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Unique URLs Count'])[0]

    # Assert
    assert_that(result['Unique URLs Count'], equal_to([0, 1, 1, 0, 5]))
    assert_that(result_none_text['Unique URLs Count'], equal_to([np.nan]))


def test_calculate_count_urls(manual_text_data_for_properties):
    # Arrange
    text_data = manual_text_data_for_properties['url_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['URLs Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['URLs Count'])[0]

    # Assert
    assert_that(result['URLs Count'], equal_to([0, 1, 1, 0, 6]))
    assert_that(result_none_text['URLs Count'], equal_to([np.nan]))


def test_calculate_count_unique_email_addresses(manual_text_data_for_properties):
    # Arrange
    text_data = manual_text_data_for_properties['email_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Unique Email Addresses Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Unique Email Addresses Count'])[0]

    # Assert
    assert_that(result['Unique Email Addresses Count'], equal_to([2, 2, 1, 0, 2, 2]))
    assert_that(result_none_text['Unique Email Addresses Count'], equal_to([np.nan]))


def test_calculate_count_email_addresses(manual_text_data_for_properties):
    # Arrange
    text_data = manual_text_data_for_properties['email_data']

    # Act
    result = calculate_builtin_properties(text_data, include_properties=['Email Addresses Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Email Addresses Count'])[0]

    # Assert
    assert_that(result['Email Addresses Count'], equal_to([2, 2, 1, 0, 2, 3]))
    assert_that(result_none_text['Email Addresses Count'], equal_to([np.nan]))


def test_calculate_count_unique_syllables(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Unique Syllables Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Unique Syllables Count'])[0]

    # Assert
    assert_that(result['Unique Syllables Count'][0: 10], equal_to([15, 11, 9, 21, 13, 17, np.nan, 8, 20, 18]))
    assert_that(result_none_text['Unique Syllables Count'], equal_to([np.nan]))


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
    assert_that(result_none_text['Reading Time'], equal_to([np.nan]))


def test_calculate_sentence_length(tweet_emotion_train_test_textdata):
    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_builtin_properties(test_text, include_properties=['Sentences Count'])[0]
    result_none_text = calculate_builtin_properties([None], include_properties=['Sentences Count'])[0]

    # Assert
    assert_that(result['Sentences Count'][0: 10], equal_to([3, 2, 1, 2, 2, 1, np.nan, 1, 2, 3]))
    assert_that(result_none_text['Sentences Count'], equal_to([np.nan]))


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


def test_calcualte_english_text_property():
    data = ['This is a sentence in English.', 'Это предложение на русском языке.']
    result = calculate_builtin_properties(data, include_properties=['English Text'])[0]
    assert_that(result['English Text'], equal_to([True, False]))


def test_calcualte_english_text_property_without_language_precalculation():
    data = ['This is a sentence in English.', 'Это предложение на русском языке.']
    assert_that([english_text(data[0]), english_text(data[1])], equal_to([True, False]))


def test_include_properties():
    # Arrange
    test_text = ['This is simple sentence.']
    # Also check capitalization doesn't matter:
    expected_properties = ['Text Length', 'Average Word Length']
    include_properties = ['Text length', 'average Word length']

    # Act
    result = calculate_builtin_properties(test_text, include_properties=include_properties)[0]
    # Assert
    for prop in result:
        assert_that(expected_properties, has_item(prop))

    # Check that raises if include_properties is not a list:
    assert_that(calling(calculate_builtin_properties).with_args(test_text, include_properties='bla'),
                raises(DeepchecksValueError))

    # Check that raises if property doesn't exist:
    assert_that(
        calling(calculate_builtin_properties).with_args(test_text, include_properties=['Non Existent Property']),
        raises(DeepchecksValueError,
               r'include_properties contains properties that were not found: \[\'non existent property\'\].'))


def test_ignore_properties():
    # Arrange
    test_text = ['This is simple sentence.']
    expected_properties = ['Text Length', 'Average Word Length', 'Max Word Length',
                           '% Special Characters', '% Punctuation', 'Language', 'Sentiment', 'Subjectivity',
                           'Lexical Density', 'Reading Ease', 'Average Words Per Sentence']

    # Also check capitalization doesn't matter:
    ignore_properties = ['Unique noun Count', 'toxicity', 'fluency', 'Formality']

    # Act
    result = calculate_builtin_properties(test_text, ignore_properties=ignore_properties)[0]
    # Assert
    for prop in result:
        assert_that(expected_properties, has_item(prop))

    # Check that raises if ignore_properties is not a list:
    assert_that(calling(calculate_builtin_properties).with_args(test_text, ignore_properties='bla'),
                raises(DeepchecksValueError))

    # Check that raises if property doesn't exist:
    assert_that(calling(calculate_builtin_properties).with_args(test_text, ignore_properties=['Non Existent Property']),
                raises(DeepchecksValueError,
                       r'ignore_properties contains properties that were not found: \[\'non existent property\'\].'))


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_properties_models_download():
    # Arrange
    model_name = 'SkolkovoInstitute/roberta_toxicity_classifier'
    model_path = MODELS_STORAGE / str(uuid.uuid4()) / model_name

    # Act
    model_download_time = timeit.timeit(
        stmt='fn()',
        number=1,
        globals={'fn': lambda: _get_transformer_model_and_tokenizer(
            property_name='',
            model_name=model_name,
            models_storage=model_path,
            use_onnx_model=False
        )}
    )

    # Assert
    assert MODELS_STORAGE.exists() and MODELS_STORAGE.is_dir()
    assert model_path.exists() and model_path.is_dir()

    # Act
    model_creation_time = timeit.timeit(
        stmt='fn()',
        number=1,
        globals={'fn': lambda: _get_transformer_model_and_tokenizer(
            property_name='',
            model_name=model_name,
            models_storage=model_path,
            use_onnx_model=False
        )}
    )

    # Assert
    assert model_creation_time <= model_download_time * 0.1


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_properties_models_download_onnx():
    directory = pathlib.Path(__file__).absolute().parent / '.nlp-models'
    model_name = TOXICITY_MODEL_NAME_ONNX
    model_path = directory / model_name

    # Act
    _get_transformer_model_and_tokenizer(property_name='', model_name=model_name, use_onnx_model=True)

    # Assert
    assert directory.exists() and directory.is_dir()
    assert model_path.exists() and model_path.is_dir()


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_batch_only_properties_calculation_with_single_samples_onnx():
    # Arrange
    text = ['Explicit is better than implicit']

    # Act
    properties, properties_types = calculate_builtin_properties(
        raw_text=text, batch_size=1,
        include_properties=['Formality', 'Text Length', 'Toxicity', 'Fluency'],
        use_onnx_models=True
    )

    # Assert
    assert_that(properties, has_entries({
        'Formality': contains_exactly(close_to(0.955, 0.01)),
        'Toxicity': contains_exactly(close_to(0.01, 0.01)),
        'Fluency': contains_exactly(close_to(0.96, 0.01)),
        'Text Length': contains_exactly(*[len(it) for it in text]),
    }))  # type: ignore


@pytest.mark.skipif(
    'TEST_NLP_PROPERTIES_MODELS_DOWNLOAD' not in os.environ,
    reason='The test takes too long to run, provide env var if you want to run it.'
)
def test_batch_only_properties_calculation_with_single_samples():
    # Arrange
    text = ['Explicit is better than implicit']

    # Act
    properties, properties_types = calculate_builtin_properties(
        raw_text=text, batch_size=1,
        include_properties=['Formality', 'Text Length', 'Toxicity', 'Fluency'],
        use_onnx_models=False
    )

    # Assert
    assert_that(properties, has_entries({
        'Formality': contains_exactly(close_to(0.955, 0.01)),
        'Toxicity': contains_exactly(close_to(0.01, 0.01)),
        'Fluency': contains_exactly(close_to(0.96, 0.01)),
        'Text Length': contains_exactly(*[len(it) for it in text]),
    }))  # type: ignore


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


def test_english_only_properties_calculated_for_all_samples():
    # Arrange
    text = [
        'Explicit is better than implicit',
        'Сьогодні чудова погода',
        'London is the capital of Great Britain'
    ]
    # Act
    properties, properties_types = calculate_builtin_properties(
        raw_text=text,
        include_properties=['Sentiment', 'Language', 'Text Length'],
        ignore_non_english_samples_for_english_properties=False
    )
    # Assert
    assert_that(properties, has_entries({
        'Sentiment': contains_exactly(close_to(0.5, 0.01), close_to(0.0, 0.01), close_to(0.8, 0.01)),
        'Language': contains_exactly('en', 'uk', 'en'),
        'Text Length': contains_exactly(*[len(it) for it in text]),
    }))  # type: ignore
    assert_that(properties_types, has_entries({
        'Sentiment': 'numeric',
        'Language': 'categorical',
        'Text Length': 'numeric',
    }))  # type: ignore


def test_sample_for_property():
    s = 'all the single ladies. all the single ladies? now put your hands up.'
    sample_words = _sample_for_property(text=s, mode='words', limit=2, random_seed=42)
    sample_sentences = _sample_for_property(text=s, mode='sentences', limit=2, random_seed=42)

    assert_that(sample_words, equal_to('hands put'))
    assert_that(sample_sentences, equal_to('all the single ladies. all the single ladies?'))
