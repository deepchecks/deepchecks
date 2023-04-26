import typing as t

import pytest
import pandas as pd
from hamcrest import *

from deepchecks.core.check_result import CheckResult
from deepchecks.nlp.checks import TextDuplicates
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent

from tests.base.utils import equal_condition_result


class DatasetWithDuplicatesInfo(t.NamedTuple):
    dataset: TextData
    n_of_duplicates_variations: int
    percent_of_duplicates: float
    duplicates_ids: t.Sequence[t.Any]
    duplicates: t.Sequence[str]


@pytest.fixture
def dataset_with_duplicates() -> DatasetWithDuplicatesInfo:
    return DatasetWithDuplicatesInfo(
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
        n_of_duplicates_variations=2,
        percent_of_duplicates=0.19999999999999996,
        duplicates_ids=(
            0, 4, 5, 9
        ),
        duplicates=(
            "Explicit is better than implicit.",
            "Readability counts.",
            "Readability counts to.",
            "Explicit, is better than implicit!",
        )
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


def test_without_duplicates(dataset_without_duplicates: TextData):
    # Arrange
    check = TextDuplicates().add_condition_ratio_less_or_equal()

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


def test_with_duplicates(dataset_with_duplicates: DatasetWithDuplicatesInfo):
    # Arrange
    dataset = dataset_with_duplicates.dataset
    check = TextDuplicates().add_condition_ratio_less_or_equal()

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
    assert_result_dataframe(
        result.value['duplicates'],
        n_of_duplicate_variations=dataset_with_duplicates.n_of_duplicates_variations,
        samples_ids=dataset_with_duplicates.duplicates_ids,
        text_samples=dataset_with_duplicates.duplicates
    )

    assert_display(display=result.display)


def assert_result_dataframe(
    df: pd.DataFrame,
    n_of_duplicate_variations: int,
    samples_ids: t.Sequence[t.Any],
    text_samples: t.Sequence[str]
):
    assert_that(df, instance_of(pd.DataFrame))
    assert_that(df.index.names, equal_to(["Duplicate", "Sample ID"]))
    assert_that(df.columns, equal_to(["Text"]))

    assert_that(
        len(frozenset(df.index.get_level_values(0))),
        equal_to(n_of_duplicate_variations)
    )
    assert_that(
        frozenset(df.index.get_level_values(1).to_list()),
        equal_to(frozenset(samples_ids))
    )
    assert_that(
        frozenset(df['Text'].to_list()),
        equal_to(frozenset(text_samples))
    )


def assert_display(display: t.Sequence[t.Any]):
    assert_that(display, has_items(
        instance_of(str),
        instance_of(str),
        instance_of(pd.DataFrame)
    )) # type: ignore

    table = t.cast(pd.DataFrame, display[2])

    assert_that(
        table.index.names,
        equal_to(["Instances", "Number of Samples"])
    )
    assert_that(
        table.columns,
        equal_to(["Text"])
    )