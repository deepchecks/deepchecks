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

import datasets

from deepchecks.core.errors import DeepchecksNotSupportedError, DeepchecksValueError
from deepchecks.nlp.task_type import TaskType

__all__ = ['TextData']

DEFAULT_LABEL_NAME = 'label'


TDataset = t.TypeVar('TDataset', bound='TextData')
TSingleLabel = t.Tuple[int, str]
TClassLabel = t.Sequence[t.Union[TSingleLabel, t.Tuple[TSingleLabel]]]
TTokenLabel = t.Sequence[t.Sequence[t.Tuple[str, int, int]]]
TTextLabel = t.Union[TClassLabel, TTokenLabel]


class TextData:
    """
    Dataset wraps HuggingFace/datasets Dataset object together with ML related metadata.

    The Dataset class contains additional data and methods intended for easily accessing
    metadata relevant for the training or validating of ML models.

    Parameters
    ----------
    df : Any
        An object that can be casted to a pandas DataFrame
         - containing data relevant for the training or validating of a ML models.
    label : t.Union[Hashable, pd.Series, pd.DataFrame, np.ndarray] , default: None
        label column provided either as a string with the name of an existing column in the DataFrame or a label
        object including the label data (pandas Series/DataFrame or a numpy array) that will be concatenated to the
        data in the DataFrame. in case of label data the following logic is applied to set the label name:
        - Series: takes the series name or 'target' if name is empty
        - DataFrame: expect single column in the dataframe and use its name
        - numpy: use 'target'
    features : t.Optional[t.Sequence[Hashable]] , default: None
        List of names for the feature columns in the DataFrame.
    cat_features : t.Optional[t.Sequence[Hashable]] , default: None
        List of names for the categorical features in the DataFrame. In order to disable categorical.
        features inference, pass cat_features=[]
    index_name : t.Optional[Hashable] , default: None
        Name of the index column in the dataframe. If set_index_from_dataframe_index is True and index_name
        is not None, index will be created from the dataframe index level with the given name. If index levels
        have no names, an int must be used to select the appropriate level by order.
    set_index_from_dataframe_index : bool , default: False
        If set to true, index will be created from the dataframe index instead of dataframe columns (default).
        If index_name is None, first level of the index will be used in case of a multilevel index.
    datetime_name : t.Optional[Hashable] , default: None
        Name of the datetime column in the dataframe. If set_datetime_from_dataframe_index is True and datetime_name
        is not None, date will be created from the dataframe index level with the given name. If index levels
        have no names, an int must be used to select the appropriate level by order.
    set_datetime_from_dataframe_index : bool , default: False
        If set to true, date will be created from the dataframe index instead of dataframe columns (default).
        If datetime_name is None, first level of the index will be used in case of a multilevel index.
    convert_datetime : bool , default: True
        If set to true, date will be converted to datetime using pandas.to_datetime.
    datetime_args : t.Optional[t.Dict] , default: None
        pandas.to_datetime args used for conversion of the datetime column.
        (look at https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html for more documentation)
    max_categorical_ratio : float , default: 0.01
        The max ratio of unique values in a column in order for it to be inferred as a
        categorical feature.
    max_categories : int , default: None
        The maximum number of categories in a column in order for it to be inferred as a categorical
        feature. if None, uses is_categorical default inference mechanism.
    label_type : str , default: None
        Used to assume target model type if not found on model. Values ('classification_label', 'regression_label')
        If None then label type is inferred from label using is_categorical logic.
    """

    _dataset: datasets.Dataset
    _task_type: t.Optional[TaskType]
    _has_label: bool
    _label_name: str
    _is_multilabel: bool = False

    def __init__(
            self,
            raw_text: t.Sequence[str],
            label: t.Optional[TTextLabel] = None,
            task_type: str = None,
            label_name: t.Optional[str] = None,
    ):

        # Require explicitly setting task type if label is provided
        if task_type is None:
            if self.label is not None:
                raise DeepchecksValueError('task_type must be set when label is provided')
            self._task_type = None
        elif task_type == 'text_classification':
            self._task_type = TaskType.TEXT_CLASSIFICATION
        elif task_type == 'token_classification':
            self._task_type = TaskType.TOKEN_CLASSIFICATION
        else:
            raise DeepchecksNotSupportedError(f'task_type {task_type} is not supported, must be one of '
                                              f'text_classification, token_classification')

        # Set label name to default if not provided
        if label_name is None:
            self._label_name = DEFAULT_LABEL_NAME
        elif isinstance(label_name, (str, float, int)):
            self._label_name = label_name
        else:
            raise DeepchecksNotSupportedError(f'label_name {label_name} is not supported, must be one of '
                                              f'str, float, int')

        self._validate_text(raw_text)
        label = self._validate_and_process_label(label, raw_text)
        # Create dataset
        self._dataset = datasets.Dataset.from_dict({'text': raw_text, self._label_name: label})
        self.encode_label()

    def encode_label(self):
        """
        Encode label as a ClassLabel object
        """
        if self._is_multilabel:
            set_of_labels = set().union(*[set(label_entry) for label_entry in self._dataset[self._label_name]])
            sorted_labels = sorted(list(set_of_labels))
            self._dataset = self._dataset.cast_column(self.label_name,
                                                      datasets.Sequence(datasets.ClassLabel(names=sorted_labels)))
        else:
            self._dataset = self._dataset.class_encode_column(self._label_name)

    @staticmethod
    def _validate_text(raw_text: t.Sequence[str]):
        """Validate text format."""
        if not isinstance(raw_text, collections.abc.Sequence):
            raise DeepchecksValueError('raw_text must be a sequence')
        if not all(isinstance(x, str) for x in raw_text):
            raise DeepchecksValueError('raw_text must be a Sequence of strings')

    def _validate_and_process_label(self, label: t.Optional[TTextLabel], raw_text: t.Sequence[str]):
        """Validate and process label to accepted formats."""
        # If label is not set, create an empty label of nulls
        if label is None:
            return [None] * len(raw_text)

        if not isinstance(label, collections.abc.Sequence):
            raise DeepchecksValueError('label must be a Sequence')

        if not len(label) == len(raw_text):
            raise DeepchecksValueError('label must be the same length as raw_text')

        if self.task_type == TaskType.TEXT_CLASSIFICATION:
            if all(isinstance(x, collections.abc.Sequence) for x in label):
                self._is_multilabel = True
                multilabel_error = 'multilabel was identified. It must be a Sequence of Sequences of ints or strings.'
                if not all(all(isinstance(y, (int, str)) for y in x) for x in label):
                    raise DeepchecksValueError(multilabel_error)
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
                if not all(
                        isinstance(x[0], str) and isinstance(x[1], int) and isinstance(x[2], int) for x in sample_label
                ):
                    raise DeepchecksValueError(token_class_error)

            # Process Token Classification label
            # TODO: Dataset can't process a tuple of different types (str, int, int). Need to convert it to something
            #       it can store

        return label

    #
    #
    # @classmethod
    # def from_numpy(
    #         cls: t.Type[TDataset],
    #         *args: np.ndarray,
    #         columns: t.Sequence[Hashable] = None,
    #         label_name: t.Hashable = None,
    #         **kwargs
    # ) -> TDataset:
    #     """Create Dataset instance from numpy arrays.
    #
    #     Parameters
    #     ----------
    #     *args: np.ndarray
    #         Numpy array of data columns, and second optional numpy array of labels.
    #     columns : t.Sequence[Hashable] , default: None
    #         names for the columns. If none provided, the names that will be automatically
    #         assigned to the columns will be: 1 - n (where n - number of columns)
    #     label_name : t.Hashable , default: None
    #         labels column name. If none is provided, the name 'target' will be used.
    #     **kwargs : Dict
    #         additional arguments that will be passed to the main Dataset constructor.
    #     Returns
    #     -------
    #     Dataset
    #         instance of the Dataset
    #     Raises
    #     ------
    #     DeepchecksValueError
    #         if receives zero or more than two numpy arrays.
    #         if columns (args[0]) is not two dimensional numpy array.
    #         if labels (args[1]) is not one dimensional numpy array.
    #         if features array or labels array is empty.
    #
    #     Examples
    #     --------
    #     >>> import numpy
    #     >>> from deepchecks.tabular import Dataset
    #
    #     >>> features = numpy.array([[0.25, 0.3, 0.3],
    #     ...                        [0.14, 0.75, 0.3],
    #     ...                        [0.23, 0.39, 0.1]])
    #     >>> labels = numpy.array([0.1, 0.1, 0.7])
    #     >>> dataset = Dataset.from_numpy(features, labels)
    #
    #     Creating dataset only from features array.
    #
    #     >>> dataset = Dataset.from_numpy(features)
    #
    #     Passing additional arguments to the main Dataset constructor
    #
    #     >>> dataset = Dataset.from_numpy(features, labels, max_categorical_ratio=0.5)
    #
    #     Specifying features and label columns names.
    #
    #     >>> dataset = Dataset.from_numpy(
    #     ...     features, labels,
    #     ...     columns=['sensor-1', 'sensor-2', 'sensor-3'],
    #     ...     label_name='labels'
    #     ... )
    #
    #     """
    #     if len(args) == 0 or len(args) > 2:
    #         raise DeepchecksValueError(
    #             "'from_numpy' constructor expecting to receive two numpy arrays (or at least one)."
    #             "First array must contains the columns and second the labels."
    #         )
    #
    #     columns_array = args[0]
    #     columns_error_message = (
    #         "'from_numpy' constructor expecting columns (args[0]) "
    #         "to be not empty two dimensional array."
    #     )
    #
    #     if len(columns_array.shape) != 2:
    #         raise DeepchecksValueError(columns_error_message)
    #
    #     if columns_array.shape[0] == 0 or columns_array.shape[1] == 0:
    #         raise DeepchecksValueError(columns_error_message)
    #
    #     if columns is not None and len(columns) != columns_array.shape[1]:
    #         raise DeepchecksValueError(
    #             f'{columns_array.shape[1]} columns were provided '
    #             f'but only {len(columns)} name(s) for them`s.'
    #         )
    #
    #     elif columns is None:
    #         columns = [str(index) for index in range(1, columns_array.shape[1] + 1)]
    #
    #     if len(args) == 1:
    #         labels_array = None
    #     else:
    #         labels_array = args[1]
    #         if len(labels_array.shape) != 1 or labels_array.shape[0] == 0:
    #             raise DeepchecksValueError(
    #                 "'from_numpy' constructor expecting labels (args[1]) "
    #                 "to be not empty one dimensional array."
    #             )
    #
    #         labels_array = pd.Series(labels_array)
    #         if label_name:
    #             labels_array = labels_array.rename(label_name)
    #
    #     return cls(
    #         df=pd.DataFrame(data=columns_array, columns=columns),
    #         label=labels_array,
    #         **kwargs
    #     )

    @property
    def dataset(self) -> datasets.Dataset:
        """Return the huggingface Dataset contain the task data."""
        return self._dataset

    #
    # def sample(self: TDataset, n_samples: int, replace: bool = False, random_state: t.Optional[int] = None,
    #            drop_na_label: bool = False) -> TDataset:
    #     """Create a copy of the dataset object, with the internal dataframe being a sample of the original dataframe.
    #
    #     Parameters
    #     ----------
    #     n_samples : int
    #         Number of samples to draw.
    #     replace : bool, default: False
    #         Whether to sample with replacement.
    #     random_state : t.Optional[int] , default None
    #         Random state.
    #     drop_na_label : bool, default: False
    #         Whether to take sample only from rows with exiting label.
    #
    #     Returns
    #     -------
    #     Dataset
    #         instance of the Dataset with sampled internal dataframe.
    #     """
    #     if drop_na_label and self.has_label():
    #         valid_idx = self.data[self.label_name].notna()
    #         data_to_sample = self.data[valid_idx]
    #     else:
    #         data_to_sample = self.data
    #     n_samples = min(n_samples, len(data_to_sample))
    #     return self.copy(data_to_sample.sample(n_samples, replace=replace, random_state=random_state))

    @property
    def n_samples(self) -> int:
        """Return number of samples in the dataset.

        Returns
        -------
        int
            Number of samples in dataset
        """
        return self._dataset.num_rows

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

    #
    # def train_test_split(self: TDataset,
    #                      train_size: t.Union[int, float, None] = None,
    #                      test_size: t.Union[int, float] = 0.25,
    #                      random_state: int = 42,
    #                      shuffle: bool = True,
    #                      stratify: t.Union[t.List, np.ndarray, bool] = False
    #                      ) -> t.Tuple[TDataset, TDataset]:
    #     """Split dataset into random train and test datasets.
    #
    #     Parameters
    #     ----------
    #     train_size : t.Union[int, float, None] , default: None
    #         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in
    #         the train split. If int, represents the absolute number of train samples. If None, the value is
    #         automatically set to the complement of the test size.
    #     test_size : t.Union[int, float] , default: 0.25
    #         If float, should be between 0.0 and 1.0 and represent the proportion of the dataset to include in the
    #         test split. If int, represents the absolute number of test samples.
    #     random_state : int , default: 42
    #         The random state to use for shuffling.
    #     shuffle : bool , default: True
    #         Whether to shuffle the data before splitting.
    #     stratify : t.Union[t.List, np.ndarray, bool] , default: False
    #         If True, data is split in a stratified fashion, using the class labels. If array-like, data is split in
    #         a stratified fashion, using this as class labels.
    #     Returns
    #     -------
    #     TextData
    #         Dataset containing train split data.
    #     TextData
    #         Dataset containing test split data.
    #     """
    #     if isinstance(stratify, bool):
    #         stratify = self.label_col if stratify else None
    #
    #     train_df, test_df = train_test_split(self._data,
    #                                          test_size=test_size,
    #                                          train_size=train_size,
    #                                          random_state=random_state,
    #                                          shuffle=shuffle,
    #                                          stratify=stratify)
    #     return self.copy(train_df), self.copy(test_df)

    @property
    def text(self) -> t.Sequence[str]:
        """Return sequence of raw text samples.

        Returns
        -------
        t.Sequence[str]
           Sequence of raw text samples.
        """
        return self._dataset['text']

    @property
    def label_name(self) -> str:
        """Return name of label.

        Returns
        -------
        str
            Name of label.
        """
        return self._label_name

    @property
    def label(self) -> TTextLabel:
        """Return the label defined in the dataset.

        Returns
        -------
        TTextLabel
        """
        return self._dataset[self.label_name]

    @property
    def classes(self) -> t.Tuple[str, ...]:
        """Return the classes from label column in list. if no label column defined, return empty list.

        Returns
        -------
        t.Tuple[str, ...]
            Classes
        """
        if self._is_multilabel:
            return self._dataset.features[self.label_name].feature.names
        return self._dataset.features[self.label_name].names

    @property
    def num_classes(self) -> int:
        """Return the number of classes from label column. if no label column defined, return 0.

        Returns
        -------
        int
            Number of classes
        """
        if self._is_multilabel:
            return self._dataset.features[self.label_name].feature.num_classes
        return self._dataset.features[self.label_name].num_classes

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
        return self.num_classes > 0

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
        if not isinstance(obj, cls.__class__):
            raise DeepchecksValueError(f'{obj} is not a {cls.__class__.__name__} instance')
        return obj

    #     if isinstance(obj, pd.DataFrame):
    #         get_logger().warning(
    #             'Received a "pandas.DataFrame" instance. It is recommended to pass a "deepchecks.tabular.Dataset" '
    #             'instance by initializing it with the data and metadata, '
    #             'for example by doing "Dataset(dataframe, label=label, cat_features=cat_features)"'
    #         )
    #         obj = Dataset(obj)
    #     elif not isinstance(obj, Dataset):
    #         raise DeepchecksValueError(
    #             f'non-empty instance of Dataset or DataFrame was expected, instead got {type(obj).__name__}'
    #         )
    #     return obj.copy(obj.data)

    @classmethod
    def datasets_share_label(cls, *datasets: 'TextData') -> bool:
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

        label_name = datasets[0].label_name
        task_type = datasets[0].task_typeF

        for ds in datasets[1:]:
            if ds.label_name != label_name:
                return False
            if ds.task_type != task_type:
                return False

        return True

    def len_when_sampled(self, n_samples: int):
        """Return number of samples in the sampled dataframe this dataset is sampled with n_samples samples."""
        return min(len(self), n_samples)

    def is_sampled(self, n_samples: int):
        """Return True if the dataset number of samples will decrease when sampled with n_samples samples."""
        return len(self) > n_samples
