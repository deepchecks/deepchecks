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

from deepchecks.nlp.utils.text_properties import MODELS_STORAGE, calculate_default_properties, get_transformer_model


def mock_fn(*args, **kwargs):  # pylint: disable=unused-argument
    return [0] * 20_000


@patch('deepchecks.nlp.utils.text_properties.run_available_kwargs', mock_fn)
def test_calculate_toxicity_property():
    # Arrange
    raw_text = ['This is a test sentence.'] * 20_000

    match_text = r'Calculating the properties \[\'Toxicity\'\] on a large dataset may take a long time.' \
                 r' Consider using a smaller sample size or running this code on better hardware. Consider using a ' \
                 r'GPU or a similar device to run these properties.'

    # Act
    with pytest.warns(UserWarning,
                      match=match_text):
        result = calculate_default_properties(raw_text, include_properties=['Toxicity'],
                                              include_long_calculation_properties=True)[0]

    # Assert
    assert_that(result, equal_to({'Toxicity': [0] * 20_000}))


def test_calculate_lexical_density_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_default_properties(test_text, include_properties=['Lexical Density'])[0]
    result_none_text = calculate_default_properties([None], include_properties=['Lexical Density'])[0]

    # Assert
    assert_that(result['Lexical Density'][0: 10], equal_to([94.44, 93.75, 100.0, 91.67, 87.5, 100.0, 100.0, 100.0, 91.67, 91.67]))
    assert_that(result_none_text['Lexical Density'], equal_to([np.nan]))


def test_calculate_unique_noun_count_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_default_properties(test_text, include_properties=['Unique Noun Count'],
                                          include_long_calculation_properties=True)[0]
    result_none_text = calculate_default_properties([None], include_properties=['Unique Noun Count'],
                                                    include_long_calculation_properties=True)[0]

    # Assert
    assert_that(result['Unique Noun Count'][0: 10], equal_to([9, 2, 3, 3, 4, 10, 4, 2, 7, 5]))
    assert_that(result_none_text['Unique Noun Count'], equal_to([np.nan]))


def test_calculate_average_sentence_length_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_default_properties(test_text, include_properties=['Average Sentence Length'])[0]
    result_none_text = calculate_default_properties([None], include_properties=['Average Sentence Length'])[0]

    # Assert
    assert_that(result['Average Sentence Length'][0: 10], equal_to([6, 7, 11, 12, 8, 19, 3, 9, 12, 7]))
    assert_that(result_none_text['Average Sentence Length'], equal_to([np.nan]))


def test_calculate_readability_score_property(tweet_emotion_train_test_textdata):

    # Arrange
    _, test = tweet_emotion_train_test_textdata
    test_text = test.text

    # Act
    result = calculate_default_properties(test_text, include_properties=['Readability Score'])[0]
    result_none_text = calculate_default_properties([None], include_properties=['Readability Score'])[0]

    # Assert
    assert_that(result['Readability Score'][0: 10], equal_to([102.0, 97.0, 80.3, 67.8, 77.1, 71.8, 91.0, 75.5, 70.1, 95.6]))
    assert_that(result_none_text['Readability Score'], equal_to([np.nan]))


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
