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
import collections
from typing import TYPE_CHECKING, Any, Dict, List, NamedTuple, Optional, Sequence, Set, Tuple, Type, cast

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksValueError, ValidationError
from deepchecks.nlp.task_type import TaskType, TTextLabel
from deepchecks.utils.logger import get_logger
from deepchecks.utils.metrics import is_label_none
from deepchecks.utils.type_inference import infer_categorical_features, infer_numerical_features
from deepchecks.utils.validation import is_sequence_not_str

if TYPE_CHECKING:
    from deepchecks.nlp.text_data import TextData


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


def label_is_null(input_label):
    """Check if the label is null for different possible input types."""
    if input_label is None:
        return True
    if is_sequence_not_str(input_label):
        if len(input_label) == 0:
            return True
        if isinstance(input_label, pd.Series):
            first_element = input_label.iloc[0]
        else:
            first_element = input_label[0]

        if is_sequence_not_str(first_element):
            return all(pd.isnull(x).all() for x in input_label)
        else:
            return all(pd.isnull(x) for x in input_label)
    else:
        return False


def validate_modify_label(labels: Optional[TTextLabel], task_type: TaskType, expected_size: int,
                          tokenized_text: Optional[Sequence[Sequence[str]]]) -> Optional[TTextLabel]:
    """Validate and process label to accepted formats."""
    if label_is_null(labels):
        return None

    if not is_sequence_not_str(labels):
        raise DeepchecksValueError('label must be a Sequence')
    if not len(labels) == expected_size:
        raise DeepchecksValueError(f'Label length ({len(labels)}) does not match expected length ({expected_size})')

    if task_type == TaskType.TEXT_CLASSIFICATION:
        if all(is_sequence_not_str(x) or is_label_none(x) for x in labels):  # Multilabel
            multilabel_error = 'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.'
            if not all(all(y in (0, 1) for y in x) for x in labels if not is_label_none(x)):
                raise DeepchecksValueError(multilabel_error)
            if any(len(labels[0]) != len(labels[i]) for i in range(len(labels)) if not is_label_none(labels[i])):
                raise DeepchecksValueError('All multilabel entries must be of the same length, which is the number'
                                           ' of possible classes.')
            labels = [[None]*len(labels[0]) if is_label_none(label_per_sample) else [int(x) for x in label_per_sample]
                      for label_per_sample in labels]
        elif any(not isinstance(x, (str, np.integer, int)) and not pd.isna(x) for x in labels):
            raise DeepchecksValueError('label must be a Sequence of strings or ints (multiclass classification) '
                                       'or a Sequence of Sequences of strings or ints (multilabel classification)')
        else:
            labels = [None if pd.isna(x) else str(x) for x in labels]
    elif task_type == TaskType.TOKEN_CLASSIFICATION:
        token_class_error = 'label must be a Sequence of Sequences of either strings or integers.'
        if not is_sequence_not_str(labels):
            raise DeepchecksValueError(token_class_error)

        result = []
        for idx, (tokens, label) in enumerate(zip(tokenized_text, labels)):  # TODO: Runs on all labels, very costly
            if is_label_none(label):
                result.append([None]*len(tokens))
            else:
                if not is_sequence_not_str(label):
                    raise DeepchecksValueError(token_class_error + f' label at {idx} was of type {type(label)}')
                if len(tokens) != len(label):
                    raise DeepchecksValueError(f'label must be the same length as tokenized_text. '
                                               f'However, for sample index {idx} received token list of length '
                                               f'{len(tokens)} and label list of length {len(label)}')
                result.append([str(x) for x in label])
        labels = result

    return np.asarray(labels, dtype=object)


class ColumnTypes(NamedTuple):
    """Utility data transfer object."""

    categorical_columns: List[str]
    numerical_columns: List[str]


def validate_length_and_type_numpy_array(data: np.ndarray, data_name: str, expected_size: int):
    """Validate length of numpy array and type."""
    if not isinstance(data, np.ndarray):
        raise DeepchecksValueError(
            f'{data_name} type {type(data)} is not supported, '
            'must be a numpy array'
        )

    if len(data) != expected_size:
        raise DeepchecksValueError(
            f'received {data_name} with {len(data)} rows, '
            f'expected {expected_size}'
        )


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
        categorical_columns = infer_categorical_features(data_table)
        get_logger().info(
            '%s types were not provided, auto inferred as categorical are:\n%s',
            data_table_name,
            categorical_columns
        )
    else:
        difference = set(categorical_columns).difference(data_table.columns)
        if len(difference) != 0:
            raise DeepchecksValueError(
                f'The following columns does not exist in {data_table_name} - {list(difference)}'
            )

    other_features = set(data_table.columns) - set(categorical_columns)
    numeric_features = infer_numerical_features(data_table[list(other_features)])

    return ColumnTypes(
        categorical_columns=list(categorical_columns),
        numerical_columns=list(numeric_features)
    )


class DataframesDifference(NamedTuple):
    """Facility type for the 'compare_dataframes' function.

    Parameters
    ==========
    only_in_train: Tuple[str, ...]
        set of columns present only in train dataframe.
    only_in_test: Tuple[str, ...]
        set of columns present only in test dataframe.
    types_mismatch: Tuple[str, ...]
        set of columns that are present in both dataframes
        but have different types.
    """

    only_in_train: Tuple[str, ...]
    only_in_test: Tuple[str, ...]
    types_mismatch: Tuple[str, ...]


class DataframesComparison(NamedTuple):
    """Facility type for the 'compare_dataframes' function.

    Parameters
    ==========
    common: Dict[str, str]
        set of columns common for both dataframes.
    difference: Optional[DataframesDifference]
        difference between two dataframes.
    """

    common: Dict[str, str]
    difference: Optional[DataframesDifference]


def compare_dataframes(
    train: pd.DataFrame,
    test: pd.DataFrame,
    train_categorical_columns: Optional[Sequence[str]] = None,
    test_categorical_columns: Optional[Sequence[str]] = None
) -> DataframesComparison:
    """Compare two dataframes and return a difference."""
    train_categorical_columns = train_categorical_columns or []
    test_categorical_columns = test_categorical_columns or []

    train_columns = cast(Set[str], set(train.columns))
    test_columns = cast(Set[str], set(test.columns))
    only_in_train = train_columns.difference(test_columns)
    only_in_test = test_columns.difference(train_columns)
    common_columns = train_columns.intersection(test_columns)
    types_mismatch: Set[str] = set()

    for column in common_columns:
        is_cat_in_both_dataframes = (
            column in train_categorical_columns
            and column in test_categorical_columns
        )

        if is_cat_in_both_dataframes:
            continue
        if not is_cat_in_both_dataframes:
            continue

        types_mismatch.add(column)

    common = {
        column: (
            'categorical'
            if column in train_categorical_columns
            else 'numerical'
        )
        for column in common_columns.difference(types_mismatch)
    }

    if only_in_train or only_in_test or types_mismatch:
        difference = DataframesDifference(
            only_in_train=tuple(only_in_train),
            only_in_test=tuple(only_in_test),
            types_mismatch=tuple(types_mismatch),
        )
    else:
        difference = None

    return DataframesComparison(common, difference)


def _validate_text_classification(
    *,
    dataset: 'TextData',
    predictions: Any = None,
    probabilities: Any = None,
    n_of_classes: Optional[int] = None,
    eps: float = 1e-3
) -> Tuple[
    Optional[np.ndarray],  # predictions
    Optional[np.ndarray],  # probabilities
]:
    if predictions is not None:
        format_error_message = (
            f'Check requires predictions for the "{dataset.name}" dataset '
            'to be of a type sequence[str] | sequence[int]'
        )
        if not is_sequence_not_str(predictions):
            raise ValidationError(format_error_message)
        if len(predictions) != dataset.n_samples:
            raise ValidationError(
                f'Check requires predictions for the "{dataset.name}" dataset '
                f'to have {dataset.n_samples} rows, same as dataset'
            )
        try:
            predictions = np.array(predictions, dtype='object')
        except ValueError as e:
            raise ValidationError(
                'Failed to cast predictions to a numpy array. '
                f'{format_error_message}'
            ) from e
        else:
            if predictions.ndim == 2 and predictions.shape[1] == 1:
                predictions = predictions[:, 0]
            if predictions.ndim != 1:
                raise ValidationError(format_error_message)

            predictions = np.array([
                str(it) if it is not None else None
                for it in predictions
            ], dtype='object')

    if probabilities is not None:
        format_error_message = (
            f'Check requires classification probabilities for the "{dataset.name}" '
            'dataset to be of a type sequence[sequence[float]] that can be cast to '
            'a 2D numpy array of shape (n_samples, n_classes)'
        )
        if len(probabilities) != dataset.n_samples:
            raise ValidationError(
                f'Check requires classification probabilities for the "{dataset.name}" '
                f'dataset to have {dataset.n_samples} rows, same as dataset'
            )
        try:
            probabilities = np.array(probabilities, dtype='float')
        except ValueError as e:
            raise ValidationError(
                'Failed to cast probabilities to a numpy array. '
                f'{format_error_message}'
            ) from e
        else:
            if len(probabilities.shape) != 2:
                raise ValidationError(format_error_message)
            if n_of_classes is not None and probabilities.shape[1] != n_of_classes:
                raise ValidationError(
                    f'Check requires classification probabilities for the "{dataset.name}" dataset '
                    f'to have {n_of_classes} columns, same as the number of classes'
                )
            if any(abs(probabilities.sum(axis=1) - 1) > eps):
                # TODO: better message
                raise ValidationError(
                    f'Check requires classification probabilities for the "{dataset.name}" '
                    f'dataset to be probabilities and sum to 1 for each row'
                )

    return predictions, probabilities


def _validate_multilabel(
    *,
    dataset: 'TextData',
    predictions: Any = None,
    probabilities: Any = None,
    n_of_classes: Optional[int] = None,
) -> Tuple[
    Optional[np.ndarray],  # predictions
    Optional[np.ndarray],  # probabilities
]:
    if predictions is not None:
        format_error_message = (
            'Check requires multi-label classification predictions for '
            f'the "{dataset.name}" dataset to be of a type sequence[sequence[int]] '
            'that can be cast to a 2D numpy array of a shape (n_samples, n_classes)'
        )
        if not is_sequence_not_str(predictions):
            raise ValidationError(format_error_message)
        if len(predictions) != dataset.n_samples:
            raise ValidationError(
                'Check requires multi-label classification predictions '
                f'for the "{dataset.name}" dataset to have {dataset.n_samples} rows, '
                'same as dataset'
            )
        try:
            predictions = np.array(predictions).astype(float)
        except ValueError as e:
            raise ValidationError(
                'Failed to cast multi-label predictions to a numpy array. '
                f'{format_error_message}'
            ) from e
        else:
            if predictions.ndim != 2:
                raise ValidationError(format_error_message)
            if n_of_classes is not None and predictions.shape[1] != n_of_classes:
                raise ValidationError(
                    'Check requires multi-label classification predictions '
                    f'for the "{dataset.name}" dataset to have {n_of_classes} columns, '
                    'same as the number of classes'
                )
            if not np.array_equal(predictions, predictions.astype(bool)):
                raise ValidationError(
                    'Check requires multi-label classification predictions '
                    f'for the "{dataset.name}" dataset to be either 0 or 1'
                )
    if probabilities is not None:
        format_error_message = (
            'Check requires multi-label classification probabilities '
            f'for the "{dataset.name}" to be of a type sequence[sequences[float]] '
            'that can be cast to a 2D numpy array of a shape (n_samples, n_classes). '
            'Each label probability value must lay between 0 and 1'
        )
        if len(probabilities) != dataset.n_samples:
            raise ValidationError(
                'Check requires multi-label classification probabilities '
                f'for the "{dataset.name}" dataset to have {dataset.n_samples} rows, '
                'same as dataset'
            )
        try:
            probabilities = np.array(probabilities, dtype='float')
        except ValueError as e:
            raise ValidationError(
                'Failed to cast multi-label probabilities to a numpy '
                f'array. {format_error_message}'
            ) from e
        else:
            if probabilities.ndim != 2:
                raise ValidationError(format_error_message)
            if n_of_classes is not None and probabilities.shape[1] != n_of_classes:
                raise ValidationError(
                    f'Check requires multi-label classification probabilities '
                    f'for the "{dataset.name}" dataset to have {n_of_classes} columns, '
                    'same as the number of classes'
                )
            if (probabilities > 1).any() or (probabilities < 0).any():
                # TODO: better message
                raise ValidationError(format_error_message)

    return predictions, probabilities


def _validate_token_classification(
    *,
    dataset: 'TextData',
    predictions: Any = None,
    probabilities: Any = None,
):
    if probabilities is not None:
        raise ValidationError(
            'For token classification probabilities are not supported'
        )

    if predictions is not None:
        format_error_message = (
            'Check requires token-classification predictions for '
            f'the "{dataset.name}" dataset to be of a type '
            'sequence[sequence[str]] or sequence[sequence[int]]'
        )
        if not is_sequence_not_str(predictions):
            raise ValidationError(format_error_message)
        if len(predictions) != dataset.n_samples:
            raise ValidationError(
                'Check requires token-classification predictions for '
                f'the "{dataset.name}" dataset to have {dataset.n_samples} rows, '
                'same as dataset'
            )

        for idx, sample_predictions in enumerate(predictions):
            if not is_sequence_not_str(sample_predictions):
                raise ValidationError(format_error_message)

            predictions_types_counter = _count_types(sample_predictions)
            criterias = (str in predictions_types_counter, int in predictions_types_counter)

            if all(criterias) or not any(criterias):
                raise ValidationError(format_error_message)

            tokenized_text = dataset.tokenized_text

            if len(sample_predictions) != len(tokenized_text[idx]):
                raise ValidationError(
                    'Check requires token-classification predictions for '
                    f'the "{dataset.name}" dataset to have the same number of tokens '
                    'as the input text'
                )


def _count_types(sequence: Sequence[Any]) -> Dict[Type, int]:
    counter = collections.defaultdict(int)
    for it in sequence:
        counter[type(it)] += 1
    return counter
