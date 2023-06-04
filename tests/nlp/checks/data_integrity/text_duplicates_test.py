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
"""Test for the NLP TextDuplicate check."""
import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.core.check_result import CheckResult
from deepchecks.nlp.checks import TextDuplicates
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result

# ====================
# ----- Fixtures -----
# ====================


class DuplicateVariation(t.NamedTuple):
    sample_ids: t.Sequence[t.Any]
    text: t.Sequence[str]


class ProblematicDataset(t.NamedTuple):
    dataset: TextData
    duplicates: t.Sequence[DuplicateVariation]
    percent_of_duplicates: float


@pytest.fixture
def dataset_with_duplicates() -> ProblematicDataset:
    return ProblematicDataset(
        dataset=TextData(raw_text=[
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Readability counts.",
            "Readability counts to.",
            "Errors should never pass silently.",
            "There should be one-- and preferably only one --obvious way to do it.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Explicit, is better than implicit!",
        ]),
        percent_of_duplicates=0.19999999999999996,
        duplicates=[
            # NOTE:
            # tests depend on items order in this list
            DuplicateVariation(
                sample_ids=[0, 9],
                text=[
                    "Explicit is better than implicit.",
                    "Explicit, is better than implicit!"
                ]
            ),
            DuplicateVariation(
                sample_ids=[4, 5],
                text=[
                    "Readability counts.",
                    "Readability counts to.",
                ]
            ),
        ],
    )


@pytest.fixture
def dataset_without_duplicates() -> TextData:
    return TextData(raw_text=[
        "Explicit is better than implicit.",
        "Simple is better than complex.",
        "Complex is better than complicated.",
        "Flat is better than nested.",
        "Readability counts.",
        "Errors should never pass silently.",
        "There should be one-- and preferably only one --obvious way to do it.",
        "If the implementation is easy to explain, it may be a good idea.",
    ])


# =================
# ----- Tests -----
# =================


def test_without_duplicates(dataset_without_duplicates: TextData):
    # Arrange
    check = TextDuplicates().add_condition_ratio_less_or_equal(0)

    # Act
    result = check.run(dataset=dataset_without_duplicates)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(
        result,
        instance_of(CheckResult)
    )
    assert_that(result.value, has_entries({
        "percent_of_duplicates": equal_to(0.0),
        "duplicates": instance_of(pd.DataFrame)
    }))
    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=True,
            name=f'Duplicate data ratio is less or equal to 0%',
            details=f'Found 0% duplicate data'
        )  # type: ignore
    )

    duplicates = t.cast(pd.DataFrame, result.value['duplicates'])
    assert_that(len(duplicates), equal_to(0))


def test_with_duplicates(dataset_with_duplicates: ProblematicDataset):
    # Arrange
    dataset = dataset_with_duplicates.dataset
    check = TextDuplicates().add_condition_ratio_less_or_equal(0)

    # Act
    result = check.run(dataset=dataset)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result, instance_of(CheckResult))

    assert_that(result.value, has_entries({
        "percent_of_duplicates": equal_to(dataset_with_duplicates.percent_of_duplicates),
        "duplicates": instance_of(pd.DataFrame)
    }))
    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=False,
            name=f'Duplicate data ratio is less or equal to {format_percent(0)}',
            details=f'Found {format_percent(dataset_with_duplicates.percent_of_duplicates)} duplicate data'
        )  # type: ignore
    )

    duplicates = result.value['duplicates']
    assert_result_dataframe(duplicates, duplicates_variations=dataset_with_duplicates.duplicates)
    assert_display(display=result.display)


# ===========================
# ----- Assertion utils -----
# ===========================


def assert_result_dataframe(
        df: pd.DataFrame,
        duplicates_variations: t.Optional[t.Sequence[DuplicateVariation]] = None,
):
    assert_that(df, instance_of(pd.DataFrame))
    assert_that(df.index.names, equal_to(["Duplicate", "Sample ID"]))
    assert_that(df.columns, equal_to(["Text"]))

    if len(df) == 0:
        return
    if not duplicates_variations:
        return

    data = df.reset_index().groupby(["Duplicate"])
    data = data.aggregate(lambda x: x.to_list())

    for idx, variant in enumerate(duplicates_variations):
        variant_data = dict(data.iloc[idx])
        assert_that(variant_data["Sample ID"], equal_to(variant.sample_ids))
        assert_that(variant_data["Text"], equal_to(variant.text))


def assert_display(display: t.Sequence[t.Any]):
    assert_that(display, has_items(
        instance_of(str),
        instance_of(str),
        instance_of(pd.io.formats.style.Styler)
    ))  # type: ignore

    table = t.cast(pd.DataFrame, display[2])
    assert_that(
        sorted(table.columns),
        equal_to(sorted(["Sample IDs", "Number of Samples", "Text"]))
    )


def test_inspect_long_samples():
    # Arrange
    dataset = TextData(raw_text=[
        ''.join(['aa'] * 500),
        ''.join(['aa'] * 500),
        ''.join(['aa'] * 600)
    ])

    check = TextDuplicates()

    # Act
    result = check.run(dataset=dataset)

    # Assert
    assert_that(result, instance_of(CheckResult))
    assert_that(result.value, has_entries({
        "percent_of_duplicates": close_to(0.33, 0.01),
        "duplicates": instance_of(pd.DataFrame)
    }))
