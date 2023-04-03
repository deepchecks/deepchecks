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
"""The dataset module containing the tabular Dataset class and its functions."""
import typing as t
import warnings
from numbers import Number

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.nlp.input_validations import (validate_length_and_calculate_column_types, validate_modify_label,
                                              validate_raw_text, validate_tokenized_text)
from deepchecks.nlp.task_type import TaskType, TTextLabel
from deepchecks.nlp.utils.text_properties import calculate_default_properties
from deepchecks.utils.logger import get_logger
from deepchecks.utils.validation import is_sequence_not_str

__all__ = ['TextData']

TDataset = t.TypeVar('TDataset', bound='TextData')


class TextData:
    """
    TextData wraps together the raw text data and the labels for the nlp task.

    The TextData class contains metadata and methods intended for easily accessing
    metadata relevant for the training or validating of ML models.

    Parameters
    ----------
    raw_text : t.Sequence[str], default: None
        The raw text data, a sequence of strings representing the raw text of each sample.
        If not given, tokenized_text must be given, and raw_text will be created from it by joining the tokens with
        spaces.
    tokenized_text : t.Sequence[t.Sequence[str]], default: None
        The tokenized text data, a sequence of sequences of strings representing the tokenized text of each sample.
        Only relevant for task_type 'token_classification'.
        If not given, raw_text must be given, and tokenized_text will be created from it by splitting the text by
        spaces.
    label : t.Optional[TTextLabel], default: None
        The label for the text data. Can be either a text_classification label or a token_classification label.
        If None, the label is not set.

        - text_classification label - For text classification the accepted label format differs between multilabel and
          single label cases. For single label data, the label should be passed as a sequence of labels, with one entry
          per sample that can be either a string or an integer. For multilabel data, the label should be passed as a
          sequence of sequences, with the sequence for each sample being a binary vector, representing the presence of
          the i-th label in that sample.
        - token_classification label - For token classification the accepted label format is the IOB format or similar
          to it. The Label must be a sequence of sequences of strings or integers, with each sequence corresponding to
          a sample in the tokenized text, and exactly the length of the corresponding tokenized text.
    task_type : str, default: None
        The task type for the text data. Can be either 'text_classification' or 'token_classification'. Must be set if
        label is provided.
    name : t.Optional[str] , default: None
        The name of the dataset. If None, the dataset name will be defined when running it within a check.
    metadata : t.Optional[pd.DataFrame] , default: None
        Metadata for the samples. If None, no metadata is set. If a DataFrame is given, it must contain
        the same number of samples as the raw_text and identical index.
    properties : t.Optional[Union[pd.DataFrame, str]] , default: None
        The text properties for the samples. If None, no properties are set. If 'auto', the properties are calculated
        using the default properties. If a DataFrame is given, it must contain the properties for each sample as the raw
        text and identical index.
    """

    _text: np.ndarray
    _label: TTextLabel
    task_type: t.Optional[TaskType]
    _tokenized_text: t.Optional[t.Sequence[t.Sequence[str]]] = None  # Outer sequence is np array
    name: t.Optional[str] = None
    _metadata: t.Optional[pd.DataFrame] = None
    _metadata_types: t.Optional[t.Dict[str, str]] = None
    _properties: t.Optional[t.Union[pd.DataFrame, str]] = None
    _properties_types: t.Optional[t.Dict[str, str]] = None
    _original_text_index: t.Optional[t.Sequence[int]] = None  # Sequence is np array

    def __init__(
            self,
            raw_text: t.Optional[t.Sequence[str]] = None,
            tokenized_text: t.Optional[t.Sequence[t.Sequence[str]]] = None,
            label: t.Optional[TTextLabel] = None,
            task_type: str = 'other',
            name: t.Optional[str] = None,
            metadata: t.Optional[pd.DataFrame] = None,
            properties: t.Optional[t.Union[pd.DataFrame]] = None,
    ):
        # Require explicitly setting task type if label is provided
        if task_type in [None, 'other']:
            if label is not None:
                raise DeepchecksValueError('task_type must be set when label is provided')
            self._task_type = TaskType.OTHER
        elif task_type == 'text_classification':
            self._task_type = TaskType.TEXT_CLASSIFICATION
        elif task_type == 'token_classification':
            if tokenized_text is None:
                raise DeepchecksValueError('tokenized_text must be provided for token_classification task type')
            validate_tokenized_text(tokenized_text)
            modified = [[str(token) for token in tokens_per_sample] for tokens_per_sample in tokenized_text]
            self._tokenized_text = np.asarray(modified, dtype=object)
            self._task_type = TaskType.TOKEN_CLASSIFICATION
        else:
            raise DeepchecksNotSupportedError(f'task_type {task_type} is not supported, must be one of '
                                              f'text_classification, token_classification, other')

        if raw_text is None and tokenized_text is None:
            raise DeepchecksValueError('Either raw_text or tokenized_text must be provided')
        elif raw_text is None:
            self._text = np.asarray([' '.join(tokens) for tokens in tokenized_text])  # Revisit this decision
        else:
            validate_raw_text(raw_text)
            self._text = np.asarray([str(x) for x in raw_text])
            if tokenized_text is not None and len(raw_text) != len(tokenized_text):
                raise DeepchecksValueError('raw_text and tokenized_text sequences must have the same length')

        self._label = validate_modify_label(label, self._task_type, len(self), tokenized_text)

        if name is not None and not isinstance(name, str):
            raise DeepchecksNotSupportedError(f'name must be a string, got {type(name)}')
        self.name = name

        self.set_metadata(metadata)
        self.set_properties(properties)

        # Used for display purposes
        self._original_text_index = np.arange(len(self))

    def is_multi_label_classification(self) -> bool:
        """Check if the dataset is multi-label."""
        if self.task_type == TaskType.TEXT_CLASSIFICATION and self._label is not None:
            return is_sequence_not_str(self._label[0])
        return False

    def copy(self: TDataset, rows_to_use: t.Optional[t.Sequence[int]] = None) -> TDataset:
        """Create a copy of this Dataset with new data.

        Parameters
        ----------
        rows_to_use : t.Optional[t.List[int]] , default: None
            The rows to use in the new copy. If None, the new copy will contain all the rows.
        """
        cls = type(self)
        logger_state = get_logger().disabled
        get_logger().disabled = True  # Make sure we won't get the warning for setting class in the non multilabel case
        if rows_to_use is None:
            new_copy = cls(raw_text=self._text, tokenized_text=self._tokenized_text, label=self._label,
                           task_type=self._task_type.value, name=self.name)
            metadata, properties = self._metadata, self._properties
            index_kept = self._original_text_index
        else:
            if not isinstance(rows_to_use, t.Sequence) or any(not isinstance(x, Number) for x in rows_to_use):
                raise DeepchecksValueError('rows_to_use must be a list of integers')
            rows_to_use = sorted(rows_to_use)
            new_copy = cls(raw_text=self._text[rows_to_use],
                           tokenized_text=self._tokenized_text[
                               rows_to_use] if self._tokenized_text is not None else None,
                           label=self._label[rows_to_use] if self.has_label() else None,
                           task_type=self._task_type.value, name=self.name)
            metadata = self._metadata.iloc[rows_to_use, :] if self._metadata is not None else None
            properties = self._properties.iloc[rows_to_use, :] if self._properties is not None else None
            index_kept = self._original_text_index[rows_to_use]

        new_copy.set_metadata(metadata, self._metadata_types)
        new_copy.set_properties(properties, self._properties_types)
        new_copy._original_text_index = index_kept  # pylint: disable=protected-access
        get_logger().disabled = logger_state
        return new_copy

    def sample(self: TDataset, n_samples: int, replace: bool = False, random_state: t.Optional[int] = None,
               drop_na_label: bool = False) -> TDataset:
        """Create a copy of the dataset object, with the internal data being a sample of the original data.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw.
        replace : bool, default: False
            Whether to sample with replacement.
        random_state : t.Optional[int] , default None
            Random state.
        drop_na_label : bool, default: False
            Whether to take sample only from rows with exiting label.

        Returns
        -------
        Dataset
            instance of the Dataset with sampled internal dataframe.
        """
        samples = np.arange(len(self))
        if drop_na_label and self.has_label():
            samples = samples[pd.notnull(self._label)]
        n_samples = min(n_samples, len(samples))

        np.random.seed(random_state)
        sample_idx = np.random.choice(range(len(samples)), n_samples, replace=replace)
        return self.copy(rows_to_use=sorted(sample_idx))

    def __len__(self) -> int:
        """Return number of samples in the dataset."""
        return self.n_samples

    @property
    def n_samples(self) -> int:
        """Return number of samples in the dataset."""
        if self._text is not None:
            return len(self._text)
        elif self._label is not None:
            return len(self._label)
        else:
            return 0

    @property
    def metadata(self) -> pd.DataFrame:
        """Return the metadata of for the dataset."""
        if self._metadata is None:
            raise DeepchecksValueError('Metadata does not exist, please set it first using the set_metadata method')
        return self._metadata

    @property
    def metadata_types(self) -> t.Dict[str, str]:
        """Return the metadata types of for the dataset."""
        if self._metadata_types is None:
            raise DeepchecksValueError('Metadata does not exist, please set it first using the set_metadata method')
        return self._metadata_types

    def set_metadata(self, metadata: pd.DataFrame, metadata_types: t.Optional[t.Dict[str, str]] = None):
        """Set metadata for the dataset.

        Parameters
        ----------
        metadata : pd.DataFrame
            Metadata of the provided textual samples.
        metadata_types : t.Optional[t.Dict[str, str]] , default : None
            The types of the metadata columns. Can be either 'numeric' or 'categorical'.
            If not provided, will be inferred automatically.
        """
        if self._metadata is not None:
            warnings.warn('Metadata already exist, overwriting it', UserWarning)

        self._metadata_types = validate_length_and_calculate_column_types(metadata, 'Metadata',
                                                                          len(self), metadata_types)
        self._metadata = metadata.reset_index(drop=True) if isinstance(metadata, pd.DataFrame) else None

    def set_properties(self, properties: pd.DataFrame, properties_types: t.Optional[t.Dict[str, str]] = None):
        """Set properties for the dataset.

        Parameters
        ----------
        properties : pd.DataFrame
            Properties of the provided textual samples.
        properties_types : t.Optional[t.Dict[str, str]] , default : None
            The types of the properties columns. Can be either 'numeric' or 'categorical'.
            If not provided, will be inferred automatically.
        """
        if self._properties is not None:
            warnings.warn('Properties already exist, overwriting it', UserWarning)

        self._properties_types = validate_length_and_calculate_column_types(properties, 'Properties',
                                                                            len(self), properties_types)
        self._properties = properties.reset_index(drop=True) if isinstance(properties, pd.DataFrame) else None

    def calculate_default_properties(self, include_properties: t.List[str] = None,
                                     ignore_properties: t.List[str] = None,
                                     include_long_calculation_properties: t.Optional[bool] = False,
                                     device: t.Optional[str] = None):
        """Calculate the default properties of the dataset.

        Parameters
        ----------
        include_properties : List[str], default None
            The properties to calculate. If None, all default properties will be calculated. Cannot be used together
            with ignore_properties parameter.
        ignore_properties : List[str], default None
            The properties to ignore. If None, no properties will be ignored. Cannot be used together with
            properties parameter.
        include_long_calculation_properties : bool, default False
            Whether to include properties that may take a long time to calculate. If False, these properties will be
            ignored.
        device : int, default None
            The device to use for the calculation. If None, the default device will be used.
        """
        if self._properties is not None:
            warnings.warn('Properties already exist, overwriting them', UserWarning)

        properties, properties_types = calculate_default_properties(
            self.text, include_properties=include_properties, ignore_properties=ignore_properties,
            include_long_calculation_properties=include_long_calculation_properties, device=device)
        self._properties = pd.DataFrame(properties, index=self.get_original_text_indexes())
        self._properties_types = properties_types

    @property
    def properties(self) -> pd.DataFrame:
        """Return the properties of the dataset."""
        if self._properties is None:
            raise DeepchecksValueError('TextData does not contain properties, add them by using '
                                       'calculate_default_properties or set_properties functions')
        return self._properties

    @property
    def properties_types(self) -> t.Dict[str, str]:
        """Return the property types of the dataset."""
        if self._properties is None:
            raise DeepchecksValueError('Properties does not exist, please set it first using the set_properties method')
        return self._properties_types

    @property
    def task_type(self) -> t.Optional[TaskType]:
        """Return the task type.

        Returns
        -------
        t.Optional[TaskType]
            Task type
        """
        return self._task_type

    @property
    def text(self) -> t.Sequence[str]:
        """Return sequence of raw text samples.

        Returns
        -------
        t.Sequence[str]
           Sequence of raw text samples.
        """
        return self._text

    @property
    def tokenized_text(self) -> t.Sequence[t.Sequence[str]]:
        """Return sequence of tokenized text samples.

        Returns
        -------
        t.Sequence[t.Sequence[str]]
           Sequence of tokenized text samples.
        """
        if self._tokenized_text is None:
            raise DeepchecksValueError('Tokenized text is not set, provide it when initializing the TextData object '
                                       'to run the requested functionalities')
        return self._tokenized_text

    @property
    def label(self) -> TTextLabel:
        """Return the label defined in the dataset.

        Returns
        -------
        TTextLabel
        """
        if not self.has_label():
            raise DeepchecksValueError('Label is not set, provide it when initializing the TextData object '
                                       'to run the requested functionalities')
        return self._label

    def has_label(self) -> bool:
        """Return True if label was set.

        Returns
        -------
        bool
           True if label was set.
        """
        return self._label is not None

    def get_original_text_indexes(self) -> t.Sequence[int]:
        """Return the original indexes of the text samples.

        Returns
        -------
        t.Sequence[int]
           Original indexes of the text samples.
        """
        return self._original_text_index

    @classmethod
    def cast_to_dataset(cls, obj: t.Any) -> 'TextData':
        """Verify Dataset or transform to Dataset.

        Function verifies that provided value is a non-empty instance of Dataset,
        otherwise raises an exception, but if the 'cast' flag is set to True it will
        also try to transform provided value to the Dataset instance.

        Parameters
        ----------
        obj
            value to verify

        Raises
        ------
        DeepchecksValueError
            if the provided value is not a TextData instance;
            if the provided value cannot be transformed into Dataset instance;
        """
        if not isinstance(obj, cls):
            raise DeepchecksValueError(f'{obj} is not a {cls.__name__} instance')
        return obj.copy()

    def validate_textdata_compatibility(self, other_text_data: 'TextData') -> bool:
        """Verify that all provided datasets share same label name and task types.

        Parameters
        ----------
        other_text_data : TextData
            The other dataset TextData object to compare with.

        Returns
        -------
        bool
            True if provided dataset share same label name and task types.
        """
        assert other_text_data is not None
        if self.task_type != other_text_data.task_type:
            return False

        return True

    def head(self, n_samples: int = 5) -> pd.DataFrame:
        """Return a copy of the dataset as a pandas Dataframe with the first n_samples samples."""
        if n_samples > len(self):
            n_samples = len(self) - 1
        result = pd.DataFrame({'text': self.text[:n_samples]}, index=self.get_original_text_indexes()[:n_samples])
        if self.has_label():
            result['label'] = self.label[:n_samples]
        if self._tokenized_text is not None:
            result['tokenized_text'] = self.tokenized_text[:n_samples]
        if self._metadata is not None:
            result = result.join(self.metadata.loc[result.index])
        return result

    def len_when_sampled(self, n_samples: t.Optional[int]):
        """Return number of samples in the sampled dataframe this dataset is sampled with n_samples samples."""
        if n_samples is None:
            return self.n_samples
        return min(self.n_samples, n_samples)

    def is_sampled(self, n_samples: t.Optional[int]):
        """Return True if the dataset number of samples will decrease when sampled with n_samples samples."""
        if n_samples is None:
            return False
        return self.n_samples > n_samples
