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
"""Fixtures for testing the nlp package"""
import random

import pytest

from deepchecks.nlp.text_data import TextData
from nltk.corpus import movie_reviews


@pytest.fixture(scope='session')
def text_classification_dataset_mock():
    """Mock for a text classification dataset"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    [0, 0, 1],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def text_classification_string_class_dataset_mock():
    """Mock for a text classification dataset"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    ['wise', 'meh', 'meh'],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def text_multilabel_classification_dataset_mock():
    """Mock for a text classification dataset"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    [[0, 0, 1], [1, 1, 0], [0, 1, 0]],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def movie_reviews_data():
    sentences = [' '.join(x) for x in movie_reviews.sents()]
    random.seed(42)
    random.shuffle(sentences)
    split_idx = int(len(sentences)/2)
    train_data = TextData(sentences[:split_idx])
    test_data = TextData(sentences[split_idx:])
    return train_data, test_data


@pytest.fixture(scope='session')
def movie_reviews_data_positive():
    pos_sentences = [' '.join(x) for x in movie_reviews.sents(categories='pos')]
    pos_data = TextData(pos_sentences)
    return pos_data


@pytest.fixture(scope='session')
def movie_reviews_data_negative():
    neg_sentences = [' '.join(x) for x in movie_reviews.sents(categories='neg')]
    neg_data = TextData(neg_sentences)
    return neg_data
