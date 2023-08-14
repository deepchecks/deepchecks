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
"""Test for the NLP FrequentSubstrings check."""

import pytest
from hamcrest import *

from deepchecks.nlp.checks.data_integrity.frequent_substrings import FrequentSubstrings
from deepchecks.nlp.text_data import TextData
from tests.base.utils import equal_condition_result


@pytest.fixture
def dataset_without_frequent_substrings():
    return TextData(
        raw_text=[
            'Hello world',
            'Do not worry be happy',
            'Explicit is better than implicit'
        ]
    )


@pytest.fixture
def dataset_with_equal_frequent_substrings():
    return TextData(
        raw_text=[
            'Explicit is better than implicit.',
            'Do not worry be happy',
            'Explicit is better than implicit...',
            'Simple is better than complex',
            'Simple is better than',
            'He said - Do not worry be happy'
        ]
    )


@pytest.fixture
def dataset_with_similar_frequent_substrings():
    return TextData(
        raw_text=[
            'Do not worry be happy. Explicit is better than implicit.',
            'Explicit is better than implicit...',
            'Simple is better than complex',
            'Simple is better than- Do not worry be happy',
            'Explicit is better than implicit.',
            'Do not worry be happy',
            'Explicit is better than implicit...',
            'Simple is better than complex',
            'Do not worry be happy!',
            'He said - Do not worry be happy'
        ]
    )


@pytest.fixture
def dataset_with_empty_string():
    return TextData(
        raw_text=[
            ''
        ]
    )

# =================
# ----- Tests -----
# =================


def test_without_frequent_substrings(dataset_without_frequent_substrings):
    # Arrange
    min_substrings = 1
    threshold = 0.5
    check = FrequentSubstrings(min_threshold=threshold).add_condition_zero_result(min_substrings=min_substrings)

    # Act
    result = check.run(dataset=dataset_without_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details='Found 0 substrings with ratio above threshold',
        name=f'There should be not more than {min_substrings} frequent substrings'
    ))


def test_with_equal_frequent_substrings(dataset_with_equal_frequent_substrings):
    # Arrange
    min_substrings = 1
    threshold = 0.05
    check = FrequentSubstrings(min_threshold=threshold).add_condition_zero_result(min_substrings=min_substrings)

    # Act
    result = check.run(dataset=dataset_with_equal_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details='Found 3 substrings with ratio above threshold',
        name=f'There should be not more than {min_substrings} frequent substrings'
    ))


def test_with_similar_frequent_substrings(dataset_with_similar_frequent_substrings):
    # Arrange
    min_substrings = 1
    threshold = 0.05
    significant_threshold = 0.3
    check = (FrequentSubstrings(min_threshold=threshold, significant_threshold=significant_threshold).
             add_condition_zero_result(min_substrings=min_substrings))

    # Act
    result = check.run(dataset=dataset_with_similar_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details='Found 3 substrings with ratio above threshold',
        name=f'There should be not more than {min_substrings} frequent substrings'
    ))


def test_with_empty_string(dataset_with_empty_string):
    # Arrange
    min_substrings = 1
    check = (FrequentSubstrings().add_condition_zero_result())

    # Act
    result = check.run(dataset=dataset_with_empty_string)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details='Found 0 substrings with ratio above threshold',
        name=f'There should be not more than {min_substrings} frequent substrings'
    ))
