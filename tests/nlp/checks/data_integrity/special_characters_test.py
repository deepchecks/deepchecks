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
"""Test for the NLP SpecialCharacters check."""
import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.nlp.checks.data_integrity.special_characters import SpecialCharacterInfo, SpecialCharacters
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result

# ====================
# ----- Fixtures -----
# ====================


@pytest.fixture
def clean_dataset():
    return TextData(
        raw_text=[
            "Hello world",
            "Do not worry be happy",
            "Weather is fine"
        ]
    )


class ProblematicDataset(t.NamedTuple):
    dataset: TextData
    special_characters: t.Dict[str, SpecialCharacterInfo]
    total_ratio_of_samples_with_spec_chars: float


@pytest.fixture
def dataset_with_special_characters() -> ProblematicDataset:
    # TODO: refactor, reduce code redundancy
    return ProblematicDataset(
        dataset=TextData(
            raw_text=[
                "Hello world¶¶",
                "Do not worry¸ be happy·",
                "Weather is fine",
                "Readability counts·",
                "Errors should never pass silently·",
            ]
        ),
        total_ratio_of_samples_with_spec_chars=0.8,
        special_characters={
            '¶': {
                'samples_ids': [0],
                'text_example': "Hello world¶¶",
                'percent_of_samples': 0.2
            },
            '¸': {
                'samples_ids': [1],
                'text_example': "Do not worry¸ be happy·",
                'percent_of_samples': 0.2
            },
            '·': {
                'samples_ids': [1, 3, 4],
                'text_example': "Do not worry¸ be happy·",
                'percent_of_samples': 0.6
            }
        }
    )


# =================
# ----- Tests -----
# =================


def test_check_on_clean_dataset(clean_dataset: TextData):
    # Arrange
    check = SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(0)

    # Act
    result = check.run(dataset=clean_dataset)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "special_characters": has_length(0),
        "total_percent_of_samples_with_spec_chars": equal_to(0)
    }))

    assert_that(result.display, has_length(0))

    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details="No special characters with ratio above threshold found",
        name='Ratio of each special character is less or equal to 0%'
    ))  # type: ignore


def test_check_on_samples_with_special_characters(dataset_with_special_characters: ProblematicDataset):
    # Arrange
    condition_threshold = 0.3
    expected_total_ratio = dataset_with_special_characters.total_ratio_of_samples_with_spec_chars
    expected_chars = dataset_with_special_characters.special_characters

    not_passed_chars = {
        k: format_percent(v['percent_of_samples'])
        for k, v in expected_chars.items()
        if v['percent_of_samples'] > condition_threshold
    }

    dataset = dataset_with_special_characters.dataset
    check = SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(condition_threshold)

    # Act
    result = check.run(dataset=dataset)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "special_characters": expected_chars,
        "total_percent_of_samples_with_spec_chars": expected_total_ratio
    }))
    assert_display(result.display)

    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details=f'Found {len(not_passed_chars)} special characters with ratio above threshold: {not_passed_chars}',
        name=f'Ratio of each special character is less or equal to {format_percent(condition_threshold)}'
    ))  # type: ignore


def test_special_characters_whitelisting(dataset_with_special_characters: ProblematicDataset):
    # Arrange
    dataset = dataset_with_special_characters.dataset
    condition_threshold = 0.3
    expected_total_ratio = 0.4
    expected_chars = dataset_with_special_characters.special_characters.copy()
    expected_chars.pop('·')

    check = SpecialCharacters(
        special_characters_whitelist=['·', *SpecialCharacters.DEFAULT_WHILTELIST]
    ).add_condition_ratio_of_special_characters_less_or_equal(condition_threshold)

    # Act
    result = check.run(dataset=dataset)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "special_characters": expected_chars,
        "total_percent_of_samples_with_spec_chars": expected_total_ratio
    }))
    assert_display(result.display)

    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details="No special characters with ratio above threshold found",
        name='Ratio of each special character is less or equal to 30%'
    ))  # type: ignore


def test_total_ratio_of_samples_condtion(dataset_with_special_characters: ProblematicDataset):
    # Arrange
    dataset = dataset_with_special_characters.dataset
    expected_chars = dataset_with_special_characters.special_characters.copy()
    expected_total_ratio = dataset_with_special_characters.total_ratio_of_samples_with_spec_chars
    condition_threshold = 0.2
    check = SpecialCharacters().add_condition_ratio_of_samples_with_special_characters_less_or_equal(condition_threshold)

    # Act
    result = check.run(dataset=dataset)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "special_characters": expected_chars,
        "total_percent_of_samples_with_spec_chars": expected_total_ratio
    }))
    assert_display(result.display)

    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=False,
        details=f'Ratio of samples with special characters is {format_percent(expected_total_ratio)}',
        name=f'Ratio of samples with special character is less or equal to {format_percent(condition_threshold)}'
    ))  # type: ignore


# ===========================
# ----- Assertion utils -----
# ===========================


def assert_display(display: t.Sequence[t.Any]):
    assert_that(display, has_items(
        instance_of(str),
        instance_of(str),
        instance_of(pd.DataFrame)
    ))
    table = t.cast(pd.DataFrame, display[2])
    assert_that(table.index.names, equal_to(['Special Character']))
    assert_that(table.columns.to_list(), equal_to(['% of Samples With Character', 'Sample IDs', 'Text Example']))
