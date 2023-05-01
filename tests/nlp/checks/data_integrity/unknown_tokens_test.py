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
"""Test for the NLP UnknownTokens check."""
import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.nlp.checks import UnknownTokens
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result


# ====================
# ----- Fixtures -----
# ====================


@pytest.fixture
def dataset_without_unknown_tokens():
    return TextData(
        label=[0, 1, 2],
        task_type="text_classification",
        raw_text=[
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
        ]
    )


@pytest.fixture
def dataset_with_unknown_tokens():
    return TextData(
        label=[0, 1, 2],
        task_type="text_classification",
        raw_text=[
            "Explicit is better than ˚implicit.",
            "Simple is better than ∫complex.",
            "Complex is better than complicated.",
        ]
    )


# =================
# ----- Tests -----
# =================


def test_without_unknown_tokens(dataset_without_unknown_tokens):
    # Arrange
    check = UnknownTokens().add_condition_ratio_of_unknown_words_less_or_equal()

    # Act
    result = check.run(dataset=dataset_without_unknown_tokens)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "unknown_word_ratio": equal_to(0),
        "unknown_word_details": instance_of(dict),
    }))

    assert_that(conditions_decisions[0], equal_condition_result(
        is_pass=True,
        details=f'Ratio was 0%',
        name='Ratio of unknown words is less than 0%',
    ))  # type: ignore

    assert_that(len(result.value['unknown_word_details']), equal_to(0))


def test_with_unknown_tokens(dataset_with_unknown_tokens):
    # Arrange
    check = UnknownTokens().add_condition_ratio_of_unknown_words_less_or_equal()

    # Act
    result = check.run(dataset=dataset_with_unknown_tokens)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "unknown_word_ratio": close_to(0.1111, 0.0001),
        "unknown_word_details": instance_of(dict),
    }))

    assert_that(conditions_decisions[0], equal_condition_result(
        is_pass=False,
        details=f'Ratio was {format_percent(result.value["unknown_word_ratio"])}',
        name='Ratio of unknown words is less than 0%',
    ))  # type: ignore

    assert_that(len(result.value['unknown_word_details']), equal_to(2))
    assert_that(result.value['unknown_word_details'], has_entries({
        "˚implicit": has_entries({
            "ratio": close_to(0.0555, 0.0001),
            "indexes": contains_exactly(0)
        }),
        "∫complex": has_entries({
            "ratio": close_to(0.0555, 0.0001),
            "indexes": contains_exactly(1)
        }),
    }))