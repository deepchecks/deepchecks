# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The dataset module containing the tabular Dataset class and its functions."""
import collections
import typing as t
from operator import itemgetter

import numpy as np
import pandas as pd

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.nlp.task_type import TaskType

__all__ = ['TextData', 'TTokenLabel', 'TClassLabel', 'TTextLabel']

from deepchecks.utils.logger import get_logger

TDataset = t.TypeVar('TDataset', bound='TextData')
TSingleLabel = t.Tuple[int, str]
TClassLabel = t.Sequence[t.Union[TSingleLabel, t.Tuple[TSingleLabel]]]
TTokenLabel = t.Sequence[t.Sequence[t.Tuple[str, int, int]]]
TTextLabel = t.Union[TClassLabel, TTokenLabel]


class TextData:
    """
    TextData wraps together the raw text data and the labels for the nlp task.

    The TextData class contains additional data and methods intended for easily accessing
    metadata relevant for the training or validating of ML models.

    Parameters
    ----------
    raw_text : t.Sequence[str]
        The raw text data, a sequence of strings representing the raw text of each sample.
    label : t.Optional[TTextLabel], default: None
        The label for the text data. Can be either a text_classification label or a token_classification label.
        If None, the label is not set.
        - text_classification label - For text classification the accepted label format differs between multilabel and
          single label cases. For single label data, the label should be passed as a sequence of labels, with one entry
          per sample that can be either a string or an integer. For multilabel data, the label should be passed as a
          sequence of sequences, with the sequence for each sample being a binary vector, representing the presence of
          the i-th label in that sample.
        - token_classification label - For token classification the accepted label format is a sequence of sequences,
          with the inner sequence containing tuples in the following format: (class_name, span_start, span_end).
          span_start and span_end are the start and end character indices of the token within the text, as it was
          passed to the raw_text argument. Each upper level sequence contains a sequence of tokens for each sample.
    task_type : str, default: None
        The task type for the text data. Can be either 'text_classification' or 'token_classification'. Must be set if
        label is provided.
    classes:  t.Optional[t.Sequence[str]], default: None
        The class names for the multilabel text classification task. May be set to define names for the classes in
        multilabel tasks, as the label input is a binary matrix that cannot convey the class names. Length must match
        the number of classes in the label, which is the number of columns for multilabel labels. Order of classes in
        this input must match order of classes as returned by model predictions to ensure they are presented correctly.
    dataset_name : t.Optional[str], default: None
        The name of the dataset. If None, the dataset name will be defined when running it within a check.
    index: t.Optional[t.Sequence[int]], default: None
        The index of the samples. If None, the index is set to np.arange(len(raw_text)).
    """

    _text: t.Sequence[str]
    _label: TTextLabel
    index: t.Sequence[t.Any]
    _task_type: t.Optional[TaskType]
    _has_label: bool
    _is_multilabel: bool = False
    _classes: t.Optional[t.List[t.Union[str, int]]] = None
    name: t.Optional[str] = None

    def __init__(
            self,
            raw_text: t.Sequence[str],
            label: t.Optional[TTextLabel] = None,
            task_type: t.Optional[str] = None,
            classes: t.Optional[t.Sequence[str]] = None,
            dataset_name: t.Optional[str] = None,
            index: t.Optional[t.Sequence[t.Any]] = None,
    ):

        # Require explicitly setting task type if label is provided
        if task_type in [None, 'other']:
            if label is not None:
                if isinstance(label, t.Sequence):
                    if pd.notnull(label).any():
                        raise DeepchecksValueError('task_type must be set when label is provided')
                else:
                    raise DeepchecksValueError('task_type must be set when label is provided')

            self._task_type = TaskType.OTHER
        elif task_type == 'text_classification':
            self._task_type = TaskType.TEXT_CLASSIFICATION
        elif task_type == 'token_classification':
            self._task_type = TaskType.TOKEN_CLASSIFICATION
        else:
            raise DeepchecksNotSupportedError(f'task_type {task_type} is not supported, must be one of '
                                              f'text_classification, token_classification, other')

        self._validate_text(raw_text)
        self._text = raw_text
        self._validate_and_set_label(label, raw_text, classes)

        if index is None:
            index = np.arange(len(raw_text))
        elif len(index) != len(raw_text):
            raise DeepchecksValueError('index must be the same length as raw_text')

        self.index = index

        if dataset_name is not None:
            if not isinstance(dataset_name, str):
                raise DeepchecksNotSupportedError(f'dataset_name type {type(dataset_name)} is not supported, must be a'
                                                  f' str')
        self.name = dataset_name

    @staticmethod
    def _validate_text(raw_text: t.Sequence[str]):
        """Validate text format."""
        if not isinstance(raw_text, collections.abc.Sequence):
            raise DeepchecksValueError('raw_text must be a sequence')
        if not all(isinstance(x, str) for x in raw_text):
            raise DeepchecksValueError('raw_text must be a Sequence of strings')

    def _validate_and_set_label(self, label: t.Optional[TTextLabel], raw_text: t.Sequence[str],
                                classes: t.Optional[t.Sequence[str]]):
        """Validate and process label to accepted formats."""
        # If label is not set, create an empty label of nulls
        self._has_label = True
        if label is None:
            self._has_label = False
            label = [None] * len(raw_text)

        if not isinstance(label, collections.abc.Sequence):
            raise DeepchecksValueError('label must be a Sequence')

        if not len(label) == len(raw_text):
            raise DeepchecksValueError('label must be the same length as raw_text')

        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            if all((isinstance(x, collections.abc.Sequence) and not isinstance(x, str)) for x in label):
                self._is_multilabel = True
                multilabel_error = 'multilabel was identified. It must be a Sequence of Sequences of 0 or 1.'
                if not all(all(y in (0, 1) for y in x) for x in label):
                    raise DeepchecksValueError(multilabel_error)
                if any(len(label[0]) != len(label[i]) for i in range(len(label))):
                    raise DeepchecksValueError('All multilabel entries must be of the same length, which is the number'
                                               ' of classes.')
            elif not all(isinstance(x, (str, int)) for x in label):
                raise DeepchecksValueError('label must be a Sequence of strings or ints or a Sequence of Sequences'
                                           'of strings or ints (for multilabel classification)')

        if self.task_type == TaskType.TOKEN_CLASSIFICATION:
            token_class_error = 'label must be a Sequence of Sequences of (str, int, int) tuples, where the string' \
                                ' is the token label, the first int is the start of the token span in the' \
                                ' raw text and the second int is the end of the token span.'
            if not all(isinstance(x, collections.abc.Sequence) for x in label):
                raise DeepchecksValueError(token_class_error)

            for sample_label in label:
                if not all(len(x) == 3 for x in sample_label):
                    raise DeepchecksValueError(token_class_error)
                if not all(
                        isinstance(x[0], str) and isinstance(x[1], int) and isinstance(x[2], int) for x in sample_label
                ):
                    raise DeepchecksValueError(token_class_error)
                if not all(x[1] < x[2] for x in sample_label):
                    raise DeepchecksValueError('Check requires token classification labels to have '
                                               'token span start before span end')

        self._label = label

        # Validate classes argument
        if classes is not None:
            if not isinstance(classes, collections.abc.Sequence):
                raise DeepchecksValueError('classes must be a Sequence of class names')
            if not all(isinstance(x, (str, int)) for x in classes):
                raise DeepchecksValueError('classes must be a Sequence of class names that are strings or ints')
            if not self._is_multilabel:
                get_logger().warning(
                    'Classes were set for a non-multilabel task. The classes will override the classes present in the '
                    'label for displays, but the same effect can be achieved by passing the intended labels in the '
                    'label argument.'
                )

            self._classes = list(classes)

    def copy(self: TDataset, raw_text: t.Optional[t.Sequence[str]] = None, label: t.Optional[TTextLabel] = None,
             index: t.Optional[t.Sequence[int]] = None) -> TDataset:
        """Create a copy of this Dataset with new data."""
        cls = type(self)
        if raw_text is None:
            raw_text = self.text
        if label is None:
            label = self.label
        if index is None:
            index = self.index
        get_logger().disabled = True  # Make sure we won't get the warning for setting class in the non multilabel case
        new_copy = cls(raw_text, label, self._task_type.value, self.classes, self.name, index)
        get_logger().disabled = False
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
        samples = self.index
        if drop_na_label and self.has_label():
            samples = samples[pd.notnull(self._label)]
        n_samples = min(n_samples, len(samples))

        np.random.seed(random_state)
        sample_idx = np.random.choice(samples, n_samples, replace=replace)
        if len(sample_idx) > 1:
            data_to_sample = {'raw_text': list(itemgetter(*sample_idx)(self._text)),
                              'label': list(itemgetter(*sample_idx)(self._label)),
                              'index': sample_idx}
        else:
            data_to_sample = {'raw_text': [self._text[sample_idx[0]]],
                              'label': [self._label[sample_idx[0]]],
                              'index': sample_idx}
        return self.copy(**data_to_sample)

    @property
    def n_samples(self) -> int:
        """Return number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in dataset
        """
        return len(self._text)

    def __len__(self):
        """Return number of samples in the dataset."""
        return self.n_samples

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
    def label(self) -> TTextLabel:
        """Return the label defined in the dataset.

        Returns
        -------
        TTextLabel
        """
        return self._label

    @property
    def classes(self) -> t.Optional[t.List[t.Union[str, int]]]:  # Bressler - should keep or use context.model_classes?
        """Return the classes from label column in list. if no label column defined, return empty list.

        Returns
        -------
        t.Tuple[str, ...]
            Classes
        """
        if self._classes is None and self.has_label():
            if self.task_type == TaskType.TEXT_CLASSIFICATION:
                if self._is_multilabel:
                    label_set = list(range(len(self._label[0])))
                else:
                    label_set = set(self._label)
            elif self.task_type == TaskType.TOKEN_CLASSIFICATION:
                label_set = self.get_tokens(self._label)
            elif self.task_type == TaskType.OTHER:
                return None
            else:
                raise DeepchecksValueError(f'Task type {self.task_type} is not supported.')
            self._classes = sorted(list(label_set))
        return self._classes

    @staticmethod
    def get_tokens(token_annotations: t.Sequence[t.Sequence[t.Tuple[str, int, int, t.Any]]]) -> t.Set[str]:
        """Return the token strings from token classification labels or predictions."""
        tokens = set()
        for sample_annotations in token_annotations:
            for annotation in sample_annotations:
                tokens.update(annotation[0])
        return tokens

    @property
    def num_classes(self) -> int:
        """Return the number of classes from label. if no label defined, return 0.

        Returns
        -------
        int
            Number of classes
        """
        return 0 if (self.classes is None) else len(self.classes)

    @property
    def is_multilabel(self) -> bool:
        """Return True if label is multilabel.

        Returns
        -------
        bool
            True if label is multilabel.
        """
        return self._is_multilabel

    def has_label(self) -> bool:
        """Return True if label was set.

        Returns
        -------
        bool
           True if label was set.
        """
        return self._has_label

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

    @classmethod
    def datasets_share_task_type(cls, *datasets: 'TextData') -> bool:
        """Verify that all provided datasets share same label name and task types.

        Parameters
        ----------
        datasets : List[TextData]
            list of TextData to validate

        Returns
        -------
        bool
            True if all TextData share same label names and task types, otherwise False

        Raises
        ------
        AssertionError
            'datasets' parameter is not a list;
            'datasets' contains less than one dataset;
        """
        assert len(datasets) > 1, "'datasets' must contains at least two items"

        task_type = datasets[0].task_type

        for ds in datasets[1:]:
            if ds.task_type != task_type:
                return False

        return True

    def len_when_sampled(self, n_samples: int):
        """Return number of samples in the sampled dataframe this dataset is sampled with n_samples samples."""
        return min(len(self), n_samples)

    def is_sampled(self, n_samples: int):
        """Return True if the dataset number of samples will decrease when sampled with n_samples samples."""
        return len(self) > n_samples
