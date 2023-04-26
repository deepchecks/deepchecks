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
"""Test for the NLP TrainTestSamplesMix check."""
import typing as t

import pandas as pd
import pytest
from hamcrest import *

from deepchecks.nlp.checks import TrainTestSamplesMix
from deepchecks.nlp.text_data import TextData
from deepchecks.utils.strings import format_percent
from tests.base.utils import equal_condition_result


class TrainTestSplit(t.NamedTuple):
    train: TextData
    test: TextData


class DuplicateVariation(t.NamedTuple):
    train_sample_ids: t.Sequence[t.Any]
    test_sample_ids: t.Sequence[t.Any]
    train_samples: t.Sequence[str]
    test_samples: t.Sequence[str]


class ProblematicTrainTestSplit(t.NamedTuple):
    train: TextData
    test: TextData
    duplicates: t.Sequence[DuplicateVariation]
    duplicates_ration: float


@pytest.fixture
def problematic_train_test_split() -> ProblematicTrainTestSplit:
    train = TextData(
        raw_text=[
            "Explicit is better than implicit.",
            "Simple is better than complex.",
            "Complex is better than complicated.",
            "Flat is better than nested.",
            "Errors should never pass silently.",
            "There should be one-- and preferably only one --obvious way to do it.",
            "If the implementation is easy to explain, it may be a good idea.",
            "Readability counts.",
            "Readability counts to.",
        ]
    )
    test = TextData(
        raw_text=[
            "Beautiful is better than ugly.",
            "Sparse is better than dense.",
            "!! Readability counts!!!",
            "Readability counts.☕",
            ",readability counts.",
            "readability !!!!! counts.",
            "For python community, readability counts.",
            "Special cases aren't special enough to break the rules.",
            "ERRORS - should never pass silently!",
            "Now is better than never.",
        ]
    )
    return ProblematicTrainTestSplit(
        train=train,
        test=test,
        duplicates_ration=0.4,
        duplicates=[
            # NOTE:
            # tests depend on items order in this list
            DuplicateVariation(
                train_sample_ids=[7, 8],
                test_sample_ids=[2, 4, 5],
                train_samples=[
                    "Readability counts.",
                    "Readability counts to.",
                ],
                test_samples=[
                    "!! Readability counts!!!",
                    ",readability counts.",
                    "readability !!!!! counts.",
                ]
            ),
            DuplicateVariation(
                train_sample_ids=[4],
                test_sample_ids=[8],
                train_samples=["Errors should never pass silently."],
                test_samples=["ERRORS - should never pass silently!"]
            ),

        ]
    )


@pytest.fixture
def train_test_datasets() -> TrainTestSplit:
    train = TextData(
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
        ]
    )
    test = TextData(
        raw_text=[
            "Beautiful is better than ugly.",
            "Sparse is better than dense.",
            "For python community, readability counts.",
            "Special cases aren't special enough to break the rules.",
            "Now is better than never.",
            "Namespaces are one honking great idea -- let's do more of those!",
        ]
    )
    return TrainTestSplit(train=train, test=test)


def test_check_execution(train_test_datasets: TrainTestSplit):
    # Arrange
    train = train_test_datasets.train
    test = train_test_datasets.test
    check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0)

    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "ratio": 0.0,
        "duplicates": instance_of(pd.DataFrame),
    }))

    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=True,
            details='No samples mix found',
            name=(
                'Percentage of test data samples that appear '
                'in train data is less or equal to 0%'
            )
        )  # type: ignore
    )

    duplicates = result.value["duplicates"]
    assert_that(len(duplicates), equal_to(0))
    assert_result_dataframe(duplicates)


def test_check_execution_with_problematic_datasets(problematic_train_test_split: ProblematicTrainTestSplit):
    # Arrange
    expected_ratio = problematic_train_test_split.duplicates_ration
    train = problematic_train_test_split.train
    test = problematic_train_test_split.test
    check = TrainTestSamplesMix().add_condition_duplicates_ratio_less_or_equal(0.1)

    # Act
    result = check.run(train_dataset=train, test_dataset=test)
    conditions_decisions = check.conditions_decision(result)

    # Assert
    assert_that(result.value, has_entries({
        "ratio": expected_ratio,
        "duplicates": instance_of(pd.DataFrame),
    }))

    assert_that(
        conditions_decisions[0],
        equal_condition_result(
            is_pass=False,
            details=(
                'Percent of test data samples that appear in '
                f'train data: {format_percent(expected_ratio)}'
            ),
            name=(
                'Percentage of test data samples that appear '
                f'in train data is less or equal to 10%'
            )
        )  # type: ignore
    )

    duplicates = result.value["duplicates"]
    assert_result_dataframe(duplicates, problematic_train_test_split.duplicates)
    assert_display(result.display)


def assert_result_dataframe(
    df: pd.DataFrame,
    duplicates_variations: t.Optional[t.Sequence[DuplicateVariation]] = None
):
    assert_that(df.index.names, equal_to(['Duplicate', 'Dataset', 'Sample ID']))
    assert_that(df.columns, equal_to(['Text']))

    if len(df) == 0:
        return
    if not duplicates_variations:
        return

    data = df.reset_index().groupby(["Duplicate", "Dataset"])
    data = data.agg(lambda x: x.to_list())

    for idx, expected_variation in enumerate(duplicates_variations):
        train_variation = dict(data.loc[(idx, "train")])
        test_variation = dict(data.loc[(idx, "test")])

        assert_that(
            train_variation["Sample ID"],
            equal_to(expected_variation.train_sample_ids)
        )
        assert_that(
            train_variation["Text"],
            equal_to(expected_variation.train_samples)
        )
        assert_that(
            test_variation["Sample ID"],
            equal_to(expected_variation.test_sample_ids)
        )
        assert_that(
            test_variation["Text"],
            equal_to(expected_variation.test_samples)
        )


def assert_display(display: t.Sequence[t.Any]):
    assert_that(display, has_items(
        instance_of(str),
        instance_of(pd.DataFrame)
    ))
    table = display[1]
    assert_that(table.index.names, equal_to(["Train instances", "Test instances"]))
    assert_that(table.columns.to_list(), equal_to(["Test text sample", "Number of test duplicates"]))
