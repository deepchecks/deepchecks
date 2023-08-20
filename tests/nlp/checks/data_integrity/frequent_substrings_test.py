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
    min_substring_ratio = 0.5
    check = (FrequentSubstrings(min_substring_ratio=min_substring_ratio)
             .add_condition_zero_result(min_substrings=min_substrings))

    # Act
    result = check.run(dataset=dataset_without_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details='Found 0 substrings with ratio above threshold',
        name=f'No more than {min_substrings} substrings with ratio above {min_substring_ratio}'
    ))
    assert_that(result.value, equal_to({}))


def test_with_equal_frequent_substrings(dataset_with_equal_frequent_substrings):
    # Arrange
    min_substrings = 1
    min_substring_ratio = 0.05
    check = (FrequentSubstrings(min_substring_ratio=min_substring_ratio).
             add_condition_zero_result(min_substrings=min_substrings))

    # Act
    result = check.run(dataset=dataset_with_equal_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    expected_value = {'Text': {0: 'Simple is better than',
                               1: 'Explicit is better than',
                               2: 'Do not worry be happy'},
                      'Frequency': {0: 0.3333333333333333,
                                    1: 0.3333333333333333,
                                    2: 0.3333333333333333},
                      '% In data': {0: '33.33%',
                                    1: '33.33%',
                                    2: '33.33%'},
                      'Sample IDs': {0: [3, 4],
                                     1: [0, 2],
                                     2: [1, 5]},
                      'Number of Samples': {0: 2, 1: 2, 2: 2}}
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details='Found 3 substrings with ratio above threshold',
        name=f'No more than {min_substrings} substrings with ratio above {min_substring_ratio}'
    ))
    assert_result_value(result.value, expected_value)


def test_with_similar_frequent_substrings(dataset_with_similar_frequent_substrings):
    # Arrange
    min_substrings = 1
    min_substring_ratio = 0.05
    check = (FrequentSubstrings(min_substring_ratio=min_substring_ratio, significant_substring_ratio=0.2).
             add_condition_zero_result(min_substrings=1))

    # Act
    result = check.run(dataset=dataset_with_similar_frequent_substrings)
    conditions_decision = check.conditions_decision(result)

    # Assert
    expected_value = {'Text': {0: 'Do not worry be',
                               1: 'Explicit is better than',
                               2: 'Simple is better than complex'},
                      'Frequency': {0: 0.5, 1: 0.4, 2: 0.2},
                      '% In data': {0: '50%', 1: '40%', 2: '20%'},
                      'Sample IDs': {0: [0, 3, 5, 8, 9], 1: [0, 1, 4, 6], 2: [2, 7]},
                      'Number of Samples': {0: 5, 1: 4, 2: 2}}
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details='Found 3 substrings with ratio above threshold',
        name=f'No more than {min_substrings} substrings with ratio above {min_substring_ratio}'
    ))
    assert_result_value(result.value, expected_value)


def test_with_empty_string(dataset_with_empty_string):
    # Arrange
    min_substrings = 1
    min_substring_ratio = 0.05
    check = (FrequentSubstrings(min_substring_ratio=min_substring_ratio).add_condition_zero_result())

    # Act
    result = check.run(dataset=dataset_with_empty_string)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details='Found 0 substrings with ratio above threshold',
        name=f'No more than {min_substrings} substrings with ratio above {min_substring_ratio}'
    ))
    assert_that(result.value, equal_to({}))


# ===========================
# ----- Assertion utils -----
# ===========================


def assert_result_value(result_value, expected_value=None):
    assert_that(result_value, instance_of(dict))
    assert_that(result_value, has_entries({
        'Text': instance_of(dict),
        'Frequency': instance_of(dict),
        '% In data': instance_of(dict),
        'Sample IDs': instance_of(dict),
        'Number of Samples': instance_of(dict),
    }))
    if expected_value is not None:
        assert_that(len(result_value), equal_to(len(expected_value)))
        for key in result_value.keys():
            assert_that(sorted(list(result_value[key].values())),
                        equal_to(sorted(list(expected_value[key].values()))))
