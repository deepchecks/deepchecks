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
"""Module containing input validation functions."""
from typing import Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, cast

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.task_type import TaskType, TTextLabel
from deepchecks.utils.logger import get_logger
from deepchecks.utils.type_inference import infer_categorical_features
from deepchecks.utils.validation import is_sequence_not_str


def validate_tokenized_text(tokenized_text: Optional[Sequence[Sequence[str]]]):
    """Validate tokenized text format."""
    error_string = 'tokenized_text must be a Sequence of Sequences of strings'
    if not is_sequence_not_str(tokenized_text):
        raise DeepchecksValueError(error_string)
    if not all(is_sequence_not_str(x) for x in tokenized_text):
        raise DeepchecksValueError(error_string)
    if not all(isinstance(x, str) for tokens in tokenized_text for x in tokens):
        raise DeepchecksValueError(error_string)


def validate_raw_text(raw_text: Optional[Sequence[str]]):
    """Validate text format."""
    error_string = 'raw_text must be a Sequence of strings'
    if not is_sequence_not_str(raw_text):
        raise DeepchecksValueError(error_string)
    if not all(isinstance(x, str) for x in raw_text):
        raise DeepchecksValueError(error_string)


def validate_modify_label(labels: Optional[TTextLabel], task_type: TaskType, expected_size: int,
                          tokenized_text: Optional[Sequence[Sequence[str]]]) -> Optional[TTextLabel]:
    """Validate and process label to accepted formats."""
    if labels is None or is_sequence_not_str(labels) and all(x is None for x in labels):
        return None

    if not is_sequence_not_str(labels):
        raise DeepchecksValueError('label must be a Sequence')
    if not len(labels) == expected_size:
        raise DeepchecksValueError(f'Label length ({len(labels)}) does not match expected length ({expected_size})')

    if task_type == TaskType.TEXT_CLASSIFICATION:
        if all(is_sequence_not_str(x) for x in labels):  # Multilabel
            multilabel_error = 'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.'
            if not all(all(y in (0, 1) for y in x) for x in labels):
                raise DeepchecksValueError(multilabel_error)
            if any(len(labels[0]) != len(labels[i]) for i in range(len(labels))):
                raise DeepchecksValueError('All multilabel entries must be of the same length, which is the number'
                                           ' of possible classes.')
            labels = [[int(x) for x in label_per_sample] for label_per_sample in labels]
        elif not all(isinstance(x, (str, int)) for x in labels):  # Classic classification
            raise DeepchecksValueError('label must be a Sequence of strings or ints (multiclass classification) '
                                       'or a Sequence of Sequences of strings or ints (multilabel classification)')
        else:
            labels = [str(x) for x in labels]
    elif task_type == TaskType.TOKEN_CLASSIFICATION:
        token_class_error = 'label must be a Sequence of Sequences of either strings or integers.'
        if not is_sequence_not_str(labels):
            raise DeepchecksValueError(token_class_error)

        result = []
        for idx, (tokens, label) in enumerate(zip(tokenized_text, labels)):  # TODO: Runs on all labels, very costly
            if not is_sequence_not_str(label):
                raise DeepchecksValueError(token_class_error + f' label at {idx} was of type {type(label)}')
            if not len(tokens) == len(label):
                raise DeepchecksValueError(f'label must be the same length as tokenized_text. '
                                           f'However, for sample index {idx} received token list of length '
                                           f'{len(tokens)} and label list of length {len(label)}')
            result.append([str(x) for x in label])
        labels = np.asarray(result, dtype=object)

    return np.asarray(labels)


class ColumnTypes(NamedTuple):
    """Utility data transfer object."""

    categorical_columns: List[str]
    numerical_columns: List[str]


def validate_length_and_calculate_column_types(
    data_table: pd.DataFrame,
    data_table_name: str,
    expected_size: int,
    categorical_columns: Optional[Sequence[str]] = None
) -> ColumnTypes:
    """Validate length of data table and calculate column types."""
    if not isinstance(data_table, pd.DataFrame):
        raise DeepchecksValueError(
            f'{data_table_name} type {type(data_table)} is not supported, '
            'must be a pandas DataFrame'
        )

    if len(data_table) != expected_size:
        raise DeepchecksValueError(
            f'received {data_table_name} with {len(data_table)} rows, '
            f'expected {expected_size}'
        )

    if categorical_columns is None:  # TODO: Add tests
        categorical_features = infer_categorical_features(data_table)
        numeric_features = [
            c for c in data_table.columns
            if c not in categorical_features
        ]

        column_types = ColumnTypes(
            categorical_columns=categorical_features,
            numerical_columns=numeric_features
        )

        get_logger().info(
            '%s types were not provided, auto inferred types are:\n%s',
            data_table_name,
            column_types._asdict()
        )

        return column_types

    difference = set(categorical_columns).difference(data_table.columns)

    if len(difference) != 0:
        raise DeepchecksValueError(
            f'The following columns does not exist in {data_table_name} - {list(difference)}'
        )

    numeric_features = [
        c for c in data_table.columns
        if c not in categorical_columns
    ]
    return ColumnTypes(
        categorical_columns=list(categorical_columns),
        numerical_columns=numeric_features
    )


class DataframesDifference(NamedTuple):
    only_in_train: Tuple[str, ...]
    only_in_test: Tuple[str, ...]
    types_mismatch: Tuple[str, ...]
    common: Dict[str, str]


def dataframes_difference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_categorical_columns: Sequence[str],
    test_categorical_columns: Sequence[str]
) -> Optional[DataframesDifference]:
    """Compare two dataframes and return a difference."""
    train_columns = cast(Set[str], set(train.columns))
    test_columns = cast(Set[str], set(test.columns))
    only_in_train = train_columns.difference(test_columns)
    only_in_test = test_columns.difference(train_columns)
    common_columns = train_columns.intersection(test_columns)
    types_mismatch: Set[str] = set()

    for column in train_columns.intersection(test_columns):
        is_cat_in_both_dataframes = (
            column in train_categorical_columns
            and column in test_categorical_columns
        )

        if is_cat_in_both_dataframes:
            continue
        if not is_cat_in_both_dataframes:
            continue

        types_mismatch.add(column)

    if only_in_train or only_in_test or types_mismatch:
        return DataframesDifference(
            only_in_train=tuple(only_in_train),
            only_in_test=tuple(only_in_test),
            types_mismatch=tuple(types_mismatch),
            common={
                column: (
                    "categorical"
                    if column in train_categorical_columns
                    else "numerical"
                )
                for column in common_columns.difference(types_mismatch)
            }
        )






