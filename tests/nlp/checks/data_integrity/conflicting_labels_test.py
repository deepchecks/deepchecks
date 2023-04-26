import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.nlp.checks import ConflictingLabels
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result


@pytest.fixture
def dataset_without_conflicts():
    return TextData(
        label=[0, 0, 0, 1, 1, 1, 2, 0, 0, 0],
        task_type="text_classification",
        raw_text=[
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
        ]
    )


class AmbiguousDuplicatVariant(t.NamedTuple):
    labels: t.Sequence[t.Any]
    sample_ids: t.Sequence[t.Any]
    text: t.Sequence[str]


class ProblematicDataset(t.NamedTuple):
    dataset: TextData
    ambiguous_samples: t.Sequence[AmbiguousDuplicatVariant]
    ambiguous_samples_ratio: float


@pytest.fixture
def dataset_with_conflicts() -> ProblematicDataset:
    return ProblematicDataset(
        dataset=TextData(
            label=['0', '0', '0', '1', '1', '1', '2', '0', '0', '1', '0'],
            task_type="text_classification",
            raw_text=[
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
                "ERRORS, should never pass silently!!",
            ]
        ),
        ambiguous_samples_ratio=0.36363636363636365,
        ambiguous_samples=[
            AmbiguousDuplicatVariant(
                labels=['0', '1'],
                sample_ids=[0, 9],
                text=[
                    "Explicit is better than implicit.",
                    "Explicit, is better than implicit!",
                ]
            ),
            AmbiguousDuplicatVariant(
                labels=['2', '0'],
                sample_ids=[6, 10],
                text=[
                    "Errors should never pass silently.",
                    "ERRORS, should never pass silently!!",
                ]
            )
        ],
    )


def test_without_conflicting_labels(dataset_without_conflicts: TextData):
    # Arrange
    check = ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal()

    # Act
    result = check.run(dataset=dataset_without_conflicts)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "percent": equal_to(0),
        "ambiguous_samples": instance_of(pd.DataFrame),
    }))

    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=True,
            details=f'Ratio of samples with conflicting labels: 0%',
            name='Ambiguous sample ratio is less or equal to 0%',
        )  # type: ignore
    )

    assert_that(len(result.value['ambiguous_samples']),equal_to(0))
    assert_result_dataframe(result.value['ambiguous_samples'])


def test_with_conflicting_labels(dataset_with_conflicts: ProblematicDataset):
    # Arrange
    expected_ratio = dataset_with_conflicts.ambiguous_samples_ratio
    dataset = dataset_with_conflicts.dataset
    check = ConflictingLabels().add_condition_ratio_of_conflicting_labels_less_or_equal()

    # Act
    result = check.run(dataset=dataset)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "percent": equal_to(expected_ratio),
        "ambiguous_samples": instance_of(pd.DataFrame),
    }))

    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=False,
            details=f'Ratio of samples with conflicting labels: {format_percent(expected_ratio)}',
            name='Ambiguous sample ratio is less or equal to 0%',
        )  # type: ignore
    )

    ambiguous_samples = result.value["ambiguous_samples"]
    assert_result_dataframe(ambiguous_samples, dataset_with_conflicts.ambiguous_samples)
    assert_display_table(result.display)


def assert_result_dataframe(
    df: pd.DataFrame,
    duplicate_variants: t.Optional[t.Sequence[AmbiguousDuplicatVariant]] = None,
):
    assert_that(
        df.index.names,
        equal_to(["Duplicate", "Sample ID", "Label"])
    )
    assert_that(
        df.columns,
        equal_to(["Text"])
    )

    if len(df) == 0:
        return
    if duplicate_variants is None:
        return

    grouped = df.reset_index().groupby(["Duplicate"])
    data = grouped.aggregate(lambda x: x.to_list())

    for idx, variant in enumerate(duplicate_variants):
        variant_data = dict(data.loc[idx])
        assert_that(variant_data["Sample ID"], equal_to(variant.sample_ids))
        assert_that(variant_data["Label"], equal_to(variant.labels))
        assert_that(variant_data["Text"], equal_to(variant.text))


def assert_display_table(display: t.Sequence[t.Any]):
    assert_that(display, has_items(
        instance_of(str),
        instance_of(str),
        instance_of(pd.DataFrame)
    ))
    table = display[2]
    assert_that(table.index.names,equal_to(["Observed Labels", "Instances"]))
    assert_that(table.columns,equal_to(["Text"]))

