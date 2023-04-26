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
"""Test for the NLP ConflictingLabels check."""
import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.nlp.checks import SpecialCharacters
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result


@pytest.fixture
def clean_dataset():
    return TextData(
        raw_text=[
            "Hello world",
            "Do not worry be happy",
            "Weather is fine"
        ]
    )


def test_check_on_clean_dataset(clean_dataset: TextData):
    # Arrange
    check = SpecialCharacters().add_condition_ratio_of_special_characters_less_or_equal(0)

    # Act
    result = check.run(dataset=clean_dataset)
    conditions_decision = check.conditions_decision(result)

    # Assert
    assert_that(result.value, all_of(instance_of(dict), has_length(0)))
    assert_that(result.display, has_length(0))

    assert_that(conditions_decision[0], equal_condition_result(
        is_pass=True,
        details="No special characters with ratio above threshold found",
        name='Ratio of each special character is less or equal to 0%'
    ))  # type: ignore

