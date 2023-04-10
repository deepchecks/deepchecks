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
from typing import Dict, Optional, Sequence

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


def validate_length_and_type(data_table: pd.DataFrame, data_table_name: str, expected_size: int):
    if not isinstance(data_table, pd.DataFrame):
        raise DeepchecksValueError(f'{data_table_name} type {type(data_table)} is not supported, must be a'
                                   f' pandas DataFrame') #TODO add comment about 'auto' mode...
    if len(data_table) != expected_size:
        raise DeepchecksValueError(f'received metadata with {len(data_table)} rows, expected {expected_size}')


def validate_length_and_calculate_column_types(data_table: pd.DataFrame, data_table_name: str, expected_size: int,
                                               column_types: Optional[Dict[str, str]] = None) -> \
        Optional[Dict[str, str]]:
    """Validate length of data table and calculate column types."""
    if data_table is None:
        return None

    validate_length_and_type(data_table, data_table_name, expected_size)

    if column_types is None:  # TODO: Add tests
        cat_features = infer_categorical_features(data_table)
        column_types = {data_table.columns[i]: 'categorical' if data_table.columns[i] in cat_features else 'numeric'
                        for i in range(len(data_table.columns))}
        get_logger().info('%s types were not provided, auto inferred types are: ', data_table_name)
        get_logger().info(column_types)
    elif sorted(list(column_types.keys())) != sorted(list(data_table.columns)):
        raise DeepchecksValueError(f'{data_table_name} types keys must identical to {data_table_name} table columns')

    return column_types
