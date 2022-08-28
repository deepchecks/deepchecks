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
import pytest

from deepchecks.nlp.text_data import TextData


@pytest.fixture(scope='session')
def text_classification_dataset_mock():
    """Mock for a text classification dataset"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    [0, 0, 1],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def text_classification_string_class_dataset_mock():
    """Mock for a text classification dataset with string labels"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    ['wise', 'meh', 'meh'],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def text_multilabel_classification_dataset_mock():
    """Mock for a multilabel text classification dataset"""
    return TextData(['I think therefore I am', 'I am therefore I think', 'I am'],
                    [[0, 0, 1], [1, 1, 0], [0, 1, 0]],
                    task_type='text_classification')


@pytest.fixture(scope='session')
def text_token_classification_dataset_mock():
    """Mock for a token classification dataset"""
    return TextData(['Mary had a little lamb', 'Mary lives in London and Paris',
                     'How much wood can a wood chuck chuck?'],
                    [[('B-PER', 0, 4)], [('B-PER', 0, 4), ('B-GEO', 14, 20), ('B-GEO', 25, 30)], []],
                    task_type='token_classification')
