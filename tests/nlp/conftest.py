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
"""Fixtures for testing the nlp package"""
# pylint: skip-file
import random

import pytest
from datasets import load_dataset
from nltk import download as nltk_download
from nltk.corpus import movie_reviews

from deepchecks.nlp.datasets.classification import tweet_emotion
from deepchecks.nlp.text_data import TextData


@pytest.fixture(scope='function')
def text_classification_dataset_mock():
    """Mock for a text classification dataset"""
    return TextData(raw_text=['I think therefore I am', 'I am therefore I think', 'I am'],
                    label=[0, 0, 1],
                    task_type='text_classification')


@pytest.fixture(scope='function')
def tweet_emotion_train_test_textdata():
    """Tweet emotion text classification dataset"""
    train, test = tweet_emotion.load_data(data_format='TextData', as_train_test=True, include_properties=True)
    return train, test


@pytest.fixture(scope='session')
def tweet_emotion_train_test_predictions():
    """Tweet emotion text classification dataset predictions"""
    return tweet_emotion.load_precalculated_predictions(pred_format='predictions', as_train_test=True)


@pytest.fixture(scope='session')
def tweet_emotion_train_test_probabilities():
    """Tweet emotion text classification dataset probabilities"""
    return tweet_emotion.load_precalculated_predictions(pred_format='probabilities', as_train_test=True)



@pytest.fixture(scope='function')
def text_classification_string_class_dataset_mock():
    """Mock for a text classification dataset with string labels"""
    return TextData(raw_text=['I think therefore I am', 'I am therefore I think', 'I am'],
                    label=['wise', 'meh', 'meh'],
                    task_type='text_classification')


@pytest.fixture(scope='function')
def text_multilabel_classification_dataset_mock():
    """Mock for a multilabel text classification dataset"""
    return TextData(raw_text=['I think therefore I am', 'I am therefore I think', 'I am'],
                    label=[[0, 0, 1], [1, 1, 0], [0, 1, 0]],
                    task_type='text_classification')


def download_nltk_resources():
    """Download nltk resources"""
    nltk_download('movie_reviews', quiet=True)
    nltk_download('punkt', quiet=True)


@pytest.fixture(scope='session')
def movie_reviews_data():
    """Dataset of single sentence samples."""
    download_nltk_resources()
    sentences = [' '.join(x) for x in movie_reviews.sents()]
    random.seed(42)

    train_data = TextData(random.sample(sentences, k=10_000))
    test_data = TextData(random.sample(sentences, k=10_000))
    return train_data, test_data


@pytest.fixture(scope='session')
def movie_reviews_data_positive():
    """Dataset of single sentence samples labeled positive."""
    download_nltk_resources()
    random.seed(42)
    pos_sentences = [' '.join(x) for x in movie_reviews.sents(categories='pos')]
    pos_data = TextData(random.choices(pos_sentences, k=1000), name='Positive')
    return pos_data


@pytest.fixture(scope='session')
def movie_reviews_data_negative():
    """Dataset of single sentence samples labeled negative."""
    download_nltk_resources()
    random.seed(42)
    neg_sentences = [' '.join(x) for x in movie_reviews.sents(categories='neg')]
    neg_data = TextData(random.choices(neg_sentences, k=1000), name='Negative')
    return neg_data

def _tokenize_raw_text(raw_text):
    """Tokenize raw text"""
    return [x.split() for x in raw_text]

@pytest.fixture(scope='session')
def text_token_classification_dataset_mock():
    """Mock for a token classification dataset"""
    return TextData(tokenized_text=_tokenize_raw_text(['Mary had a little lamb', 'Mary lives in London and Paris',
                              'How much wood can a wood chuck chuck?']),
                    label=[['B-PER', 'O', 'O', 'O', 'O'], ['B-PER', 'O', 'O', 'B-GEO', 'O', 'B-GEO'],
                           ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']],
                    task_type='token_classification')


@pytest.fixture(scope='session')
def wikiann():
    """Wikiann dataset for token classification"""
    dataset = load_dataset('wikiann', name='en', split='train')

    data = list([' '.join(l.as_py()) for l in dataset.data['tokens']])  # pylint: disable=consider-using-generator
    ner_tags = dataset.data['ner_tags']
    ner_to_iob_dict = {0: 'O', 1: 'B-PER', 2: 'I-PER', 3: 'B-ORG', 4: 'I-ORG', 5: 'B-LOC', 6: 'I-LOC'}
    ner_tags_translated = [[ner_to_iob_dict[ner_tag] for ner_tag in ner_tag_list.as_py()] for ner_tag_list in ner_tags]

    return TextData(tokenized_text=_tokenize_raw_text(data), label=ner_tags_translated, task_type='token_classification')
