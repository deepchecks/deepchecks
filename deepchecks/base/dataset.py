# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The Dataset module containing the dataset Class and its functions."""
# pylint: disable=inconsistent-quotes

import typing as t
import logging

import numpy as np
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype

from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.utils.strings import is_string_column
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


__all__ = ['Dataset', 'ensure_dataframe_type']

logger = logging.getLogger('deepchecks.dataset')

TDataset = t.TypeVar('TDataset', bound='Dataset')


class Dataset:
    """Dataset wraps pandas DataFrame together with ML related metadata.

    The Dataset class is containing additional data and methods intended for easily accessing
    metadata relevant for the training or validating of a ML models.

    Attributes:
        _features: List of names for the feature columns in the DataFrame.
        _label: Name of the label column in the DataFrame.
        _use_index: Boolean value controlling whether the DataFrame index will be used by the index_col property.
        _index_name: Name of the index column in the DataFrame.
        _date_name: Name of the date column in the DataFrame.
        cat_features: List of names for the categorical features in the DataFrame.
    """

    _features: t.List[Hashable]
    _label_name: t.Optional[Hashable]
    _use_default_index: bool
    _index_name: t.Optional[Hashable]
    _date_name: t.Optional[Hashable]
    cat_features: t.List[Hashable]
    _data: pd.DataFrame
    _max_categorical_ratio: float
    _max_categories: int

    def __init__(
        self,
        df: pd.DataFrame,
        label: pd.Series = None,
        features: t.Optional[t.Sequence[Hashable]] = None,
        cat_features: t.Optional[t.Sequence[Hashable]] = None,
        label_name: t.Optional[Hashable] = None,
        use_default_index: bool = False,
        index_name: t.Optional[Hashable] = None,
        date_name: t.Optional[Hashable] = None,
        date_args: t.Optional[t.Dict] = None,
        convert_date_: bool = True,
        max_categorical_ratio: float = 0.01,
        max_categories: int = 30,
        max_float_categories: int = 5
    ):
        """Initiate the Dataset using a pandas DataFrame and Metadata.

        Args:
            df (pandas.DataFrame):
                A pandas DataFrame containing data relevant for the training or validating of a ML models.
            label (pandas.Series)
                A pandas series containing data of the labels. Will be joined to the data dataframe with the name
                given by `label_name` parameter or 'target' by default.
            features (Optional[Sequence[Hashable]]):
                List of names for the feature columns in the DataFrame.
            cat_features (Optional[Sequence[Hashable]]):
                List of names for the categorical features in the DataFrame. In order to disable categorical.
                features inference, pass cat_features=[]
            label_name (Optional[Hashable]):
                If `label` is given, then this name is used as the column name for the labels.
                If `label` is none, then looks for this name in the data dataframe.
            use_default_index (bool, default False):
                Whether to use the dataframe index as the index column, for index related checks. can't be used
                together with `index_name`
            index_name (Optional[Hashable]):
                Name of the index column in the DataFrame. can't be used together with `use_default_index`
            date_name (Optional[Hashable]):
                Name of the date column in the DataFrame.
            date_args (Optional[Dict]):
                pandas.to_datetime args used for conversion of the date column.
                (look at https://pandas.pydata.org/docs/reference/api/pandas.to_datetime.html for more documentation)
            max_categorical_ratio (float, default 0.01):
                The max ratio of unique values in a column in order for it to be inferred as a
                categorical feature.
            max_categories (int, default 30):
                The maximum number of categories in a column in order for it to be inferred as a categorical
                feature.
            max_float_categories (int, default 5):
                The maximum number of categories in a float column in order for it to be inferred as a
                categorical feature.
        """
        self._data = df.copy()

        # Validations
        if label is not None:
            if label.shape[0] != self._data.shape[0]:
                raise DeepchecksValueError('Number of samples of label and data must be equal')
            # Make tests to prevent overriding user column
            if label_name is None:
                label_name = 'target'
                if label_name in self._data.columns:
                    raise DeepchecksValueError(f'Data has column with name "{label_name}", use label_name parameter'
                                               'to set column name for label which does\'t exists in the data')
            else:
                if label_name in self._data.columns:
                    raise DeepchecksValueError('Can\'t pass label with label_name that exists in the data. change '
                                               'the label_name parameter')
            self._data[label_name] = label

        if use_default_index is True and index_name is not None:
            raise DeepchecksValueError('parameter use_default_index cannot be True if index is given')

        if index_name is not None and index_name not in self._data.columns:
            error_message = f'index column {index_name} not found in dataset columns.'
            if index_name == 'index':
                error_message += ' If you attempted to use the dataframe index, set use_default_index to True instead.'
            raise DeepchecksValueError(error_message)

        if date_name is not None and date_name not in self._data.columns:
            raise DeepchecksValueError(f'date column {date_name} not found in dataset columns')

        if label_name is not None and label_name not in self._data.columns:
            raise DeepchecksValueError(f'label column {label_name} not found in dataset columns')

        if features:
            difference = set(features) - set(self._data.columns)
            if len(difference) > 0:
                raise DeepchecksValueError('Features must be names of columns in dataframe. '
                                           f'Features {difference} have not been '
                                           'found in input dataframe.')
            self._features = list(features)
        else:
            self._features = [x for x in self._data.columns if x not in {label_name, index_name, date_name}]

        self._label_name = label_name
        self._use_default_index = use_default_index
        self._index_name = index_name
        self._date_name = date_name
        self._date_args = date_args or {}

        self._max_categorical_ratio = max_categorical_ratio
        self._max_categories = max_categories
        self._max_float_categories = max_float_categories

        if self._label_name in self.features:
            raise DeepchecksValueError(f'label column {self._label_name} can not be a feature column')

        if self._label_name:
            try:
                self.check_compatible_labels()
            except DeepchecksValueError as e:
                logger.warning(str(e))

        if self._date_name in self.features:
            raise DeepchecksValueError(f'date column {self._date_name} can not be a feature column')

        if self._index_name in self.features:
            raise DeepchecksValueError(f'index column {self._index_name} can not be a feature column')

        if cat_features is not None:
            if set(cat_features).intersection(set(self._features)) != set(cat_features):
                raise DeepchecksValueError(f'Categorical features must be a subset of features. '
                                           f'Categorical features {set(cat_features) - set(self._features)} '
                                           f'have not been found in feature list.')
            self.cat_features = list(cat_features)
        else:
            self.cat_features = self.infer_categorical_features()

        if self._date_name and convert_date_:
            self._data[self._date_name] = self._data[self._date_name].apply(pd.to_datetime, **self._date_args)

    @classmethod
    def from_numpy(
        cls: t.Type[TDataset],
        *args: np.ndarray,
        columns: t.Sequence[Hashable] = None,
        **kwargs
    ) -> TDataset:
        """Create Dataset instance from numpy arrays.

        Args:
            *args: (np.ndarray):
                Numpy array of data columns, and second optional numpy array of labels.
            columns (Sequence[Hashable], default None):
                names for the columns. If none provided, the names that will be automatically
                assigned to the columns will be: 1 - n (where n - number of columns)
            label_name (Hashable, default None):
                labels column name. If none is provided, the name 'target' will be used.
            **kwargs:
                additional arguments that will be passed to the main Dataset constructor.

        Returns:
            Dataset: instance of the Dataset

        Raises:
            DeepchecksValueError:
                if receives zero or more than two numpy arrays;
                if columns (args[0]) is not two dimensional numpy array;
                if labels (args[1]) is not one dimensional numpy array;
                if features array or labels array is empty;

        Examples
        --------
        >>> features = np.array([[0.25, 0.3, 0.3], [0.14, 0.75, 0.3], [0.23, 0.39, 0.1]])
        >>> labels = np.array([0.1, 0.1, 0.7])
        >>> dataset = Dataset.from_numpy(features, labels)

        Creating dataset only from features array.

        >>> dataset = Dataset.from_numpy(features)

        Passing additional arguments to the main Dataset constructor

        >>> dataset = Dataset.from_numpy(
        ...    features, labels,
        ...    max_categorical_ratio=0.5
        ... )

        Specifying features and label columns names.

        >>> dataset = Dataset.from_numpy(
        ...    features, labels,
        ...    columns=['sensor-1', 'sensor-2', 'sensor-3',],
        ...    label_name='labels'
        ... )

        """
        if len(args) == 0 or len(args) > 2:
            raise DeepchecksValueError(
                "'from_numpy' constructor expecting to receive two numpy arrays (or at least one)."
                "First array must contains the columns and second the labels."
            )

        columns_array = args[0]
        columns_error_message = (
            "'from_numpy' constructor expecting columns (args[0]) "
            "to be not empty two dimensional array."
        )

        if len(columns_array.shape) != 2:
            raise DeepchecksValueError(columns_error_message)

        if columns_array.shape[0] == 0 or columns_array.shape[1] == 0:
            raise DeepchecksValueError(columns_error_message)

        if columns is not None and len(columns) != columns_array.shape[1]:
            raise DeepchecksValueError(
                f'{columns_array.shape[1]} columns were provided '
                f'but only {len(columns)} name(s) for them`s.'
            )

        elif columns is None:
            columns = [str(index) for index in range(1, columns_array.shape[1] + 1)]

        if len(args) == 1:
            labels_array = None
        else:
            labels_array = args[1]
            if len(labels_array.shape) != 1 or labels_array.shape[0] == 0:
                raise DeepchecksValueError(
                    "'from_numpy' constructor expecting labels (args[1]) "
                    "to be not empty one dimensional array."
                )

            labels_array = pd.Series(labels_array)

        return cls(
            df=pd.DataFrame(data=columns_array, columns=columns),
            label=labels_array,
            **kwargs
        )

    @classmethod
    def from_dict(
        cls: t.Type[TDataset],
        data: t.Mapping[Hashable, t.Any],
        orient: str = 'columns',
        dtype: t.Optional[np.dtype] = None,
        columns: t.Optional[t.Sequence[Hashable]] = None,
        **kwargs
    ) -> TDataset:
        """Create instance of the Dataset from the dict object.

        Args:
            data (t.Mapping[t.Hashable, t.Any]):
                dict from which to create a dataset
            orient (Literal['columns'] | Literal['index'], default 'columns'):
                The “orientation” of the data. Will be passed to the dataframe constructor.
            dtype (Optional[numpy.dtype], default None):
                Data type to force, otherwise infer. Will be passed to the dataframe constructor.
            columns (t.Optional[t.Sequence[t.Hashable]]):
                Column labels. Will be passed to the dataframe constructor.
            **kwargs:
                additional arguments that will be passed to the main Dataset constructor.

        Returns:
            Dataset: instance of the Dataset.
        """
        return cls(
            df=pd.DataFrame.from_dict(data=data, orient=orient, dtype=dtype, columns=columns),
            **kwargs
        )

    @property
    def data(self) -> pd.DataFrame:
        """Return the data of dataset."""
        return self._data

    def copy(self: TDataset, new_data) -> TDataset:
        """Create a copy of this Dataset with new data."""
        # Filter out if columns were dropped
        features = list(set(self._features).intersection(new_data.columns))
        cat_features = list(set(self.cat_features).intersection(new_data.columns))
        label_name = self._label_name if self._label_name in new_data.columns else None
        index = self._index_name if self._index_name in new_data.columns else None
        date = self._date_name if self._date_name in new_data.columns else None

        cls = type(self)

        return cls(new_data, features=features, cat_features=cat_features, label_name=label_name,
                   use_default_index=self._use_default_index, index_name=index, date_name=date, convert_date_=False,
                   max_categorical_ratio=self._max_categorical_ratio, max_categories=self._max_categories)

    @property
    def n_samples(self) -> int:
        """Return number of samples in dataframe.

        Returns:
           Number of samples in dataframe
        """
        return self.data.shape[0]

    def __len__(self):
        """Return number of samples in the member dataframe."""
        return self.n_samples

    def infer_categorical_features(self) -> t.List[Hashable]:
        """Infers which features are categorical by checking types and number of unique values.

        Returns:
           Out of the list of feature names, returns list of categorical features
        """
        cat_columns = []

        # Checking for categorical dtypes
        cat_dtypes = self.data.select_dtypes(include='category')
        if len(cat_dtypes.columns) > 0:
            return list(cat_dtypes.columns)

        for col in self._features:
            if self.is_categorical(col):
                cat_columns.append(col)

        if len(cat_columns) > 0:
            if len(cat_columns) < 7:
                stringified_columns = ", ".join(map(str, cat_columns))
                print(
                    'Automatically inferred these columns as categorical features: '
                    f'{stringified_columns}. \n'
                )
            else:
                stringified_columns = ", ".join(list(map(str, cat_columns))[:7])
                print(
                    'Some columns have been inferred as categorical features: '
                    f'{stringified_columns}. \n and more... \n For the full list '
                    'of columns, use dataset.cat_features'
                )

        return cat_columns

    def is_categorical(self, col_name: Hashable) -> bool:
        """Check if uniques are few enough to count as categorical.

        Args:
            col_name (str): The name of the column in the dataframe

        Returns:
            If is categorical according to input numbers
        """
        col_data = self.data[col_name]
        n_unique = col_data.nunique(dropna=True)
        n_samples = len(col_data.dropna())

        if is_float_dtype(col_data):
            return n_unique <= self._max_float_categories

        return n_unique / n_samples < self._max_categorical_ratio and n_unique <= self._max_categories

    @property
    def index_name(self) -> t.Optional[Hashable]:
        """If index column exists, return its name.

        Returns:
           (str) index column name
        """
        return self._index_name

    @property
    def index_col(self) -> t.Optional[pd.Series]:
        """Return index column. Index can be a named column or DataFrame index.

        Returns:
           If date column exists, returns a pandas Series of the index column.
        """
        if self._use_default_index is True:
            return pd.Series(self.data.index)
        elif self._index_name is not None:
            return self.data[self._index_name]
        else:  # No meaningful index to use: Index column not configured, and use_column is False
            return

    @property
    def date_name(self) -> t.Optional[Hashable]:
        """If date column exists, return its name.

        Returns:
           (str) date column name
        """
        return self._date_name

    @property
    def date_col(self) -> t.Optional[pd.Series]:
        """Return date column if exists.

        Returns:
           (Series): Series of the date column
        """
        return self.data[self._date_name] if self._date_name else None

    @property
    def label_name(self) -> t.Optional[Hashable]:
        """If label column exists, return its name.

        Returns:
           (str) Label name
        """
        return self._label_name

    @property
    def label_col(self) -> t.Optional[pd.Series]:
        """Return label column if exists.

        Returns:
           Label column
        """
        return self.data[self._label_name] if self._label_name else None

    @property
    def features(self) -> t.List[Hashable]:
        """Return list of feature names.

        Returns:
           List of feature names.
        """
        return self._features

    @property
    def features_columns(self) -> t.Optional[pd.DataFrame]:
        """Return features columns if exists.

        Returns:
           Features columns
        """
        return self.data[self._features] if self._features else None

    @property
    def columns_info(self) -> t.Dict[Hashable, str]:
        """Return the role and logical type of each column.

        Returns:
           Diractory of a column and its role
        """
        columns = {}
        for column in self.data.columns:
            if column == self._index_name:
                value = 'index'
            elif column == self._date_name:
                value = 'date'
            elif column == self._label_name:
                value = 'label'
            elif column in self._features:
                if column in self.cat_features:
                    value = 'categorical feature'
                else:
                    value = 'numerical feature'
            else:
                value = 'other'
            columns[column] = value
        return columns

    def check_compatible_labels(self):
        """Check if label column is supported by deepchecks."""
        labels = self.label_col
        if labels is None:
            return
        if is_string_column(labels):
            raise DeepchecksValueError('String labels are not supported')
        elif pd.isnull(labels).any():
            raise DeepchecksValueError('Can not have null values in label column')

    # Validations:

    def validate_label(self):
        """
        Throws error if dataset does not have a label.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have a label

        """
        if self.label_name is None:
            raise DeepchecksValueError('Check requires dataset to have a label column')
        self.check_compatible_labels()

    def validate_features(self):
        """
        Throws error if dataset does not have a features columns.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError: if dataset does not have features columns.
        """
        if not self._features:
            raise DeepchecksValueError('Check requires dataset to have features columns!')

    def validate_date(self):
        """
        Throws error if dataset does not have a date column.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have a date column

        """
        if self.date_name is None:
            raise DeepchecksValueError('Check requires dataset to have a date column')

    def validate_index(self):
        """
        Throws error if dataset does not have an index column / does not use dataframe index as index.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have an index

        """
        if self.index_name is None:
            raise DeepchecksValueError('Check requires dataset to have an index column')

    def filter_columns_with_validation(
        self: TDataset,
        columns: t.Union[Hashable, t.List[Hashable], None] = None,
        ignore_columns: t.Union[Hashable, t.List[Hashable], None] = None
    ) -> TDataset:
        """Filter dataset columns by given params.

        Args:
            columns (Union[Hashable, List[Hashable], None]): Column names to keep.
            ignore_columns (Union[Hashable, List[Hashable], None]): Column names to drop.

        Returns:
            TDataset: horizontally filtered dataset

        Raise:
            DeepchecksValueError: In case one of columns given don't exists raise error
        """
        new_data = filter_columns_with_validation(self.data, columns, ignore_columns)
        if new_data.equals(self.data):
            return self
        else:
            return self.copy(new_data)

    def validate_shared_features(self, other) -> t.List[Hashable]:
        """
        Return the list of shared features if both datasets have the same feature column names. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            check_name (str): check name to print in error

        Returns:
            List[Hashable] - list of shared features names

        Raises:
            DeepchecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other)
        if sorted(self.features) == sorted(other.features):
            return self.features
        else:
            raise DeepchecksValueError('Check requires datasets to share the same features')

    def validate_shared_categorical_features(self, other) -> t.List[Hashable]:
        """
        Return list of categorical features if both datasets have the same categorical features. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            check_name (str): check name to print in error

        Returns:
            List[Hashable] - list of shared features names

        Raises:
            DeepchecksValueError if datasets don't have the same features
        """
        Dataset.validate_dataset(other)
        if sorted(self.cat_features) == sorted(other.cat_features):
            return self.cat_features
        else:
            raise DeepchecksValueError('Check requires datasets to share '
                                       'the same categorical features. Possible reason is that some columns were'
                                       'inferred incorrectly as categorical features. To fix this, manually edit the '
                                       'categorical features using Dataset(cat_features=<list_of_features>')

    def validate_shared_label(self, other) -> Hashable:
        """Verify presence of shared labels.

        Return the list of shared features if both datasets have the same
        feature column names, else, raise error.

        Args:
            other (Dataset): Expected to be Dataset type. dataset to compare
            check_name (str): check name to print in error

        Returns:
            Hashable: name of the label column

        Raises:
            DeepchecksValueError if datasets don't have the same label
        """
        Dataset.validate_dataset(other)
        if (
            self.label_name is not None and other.label_name is not None
            and self.label_name == other.label_name
        ):
            return t.cast(Hashable, self.label_name)
        else:
            raise DeepchecksValueError('Check requires datasets to share the same label')

    @classmethod
    def validate_dataset_or_dataframe(cls, obj) -> 'Dataset':
        """
        Raise error if object is not pandas DataFrame or deepcheck Dataset and returns the object as deepchecks Dataset.

        Args:
            obj: object to validate as dataset

        Returns:
            (Dataset): object converted to deepchecks dataset
        """
        if isinstance(obj, Dataset):
            if len(obj.data) == 0:
                raise DeepchecksValueError('dataset cannot be empty')
            return obj
        elif isinstance(obj, pd.DataFrame):
            if len(obj) == 0:
                raise DeepchecksValueError('dataset cannot be empty')
            return Dataset(obj)
        else:
            raise DeepchecksValueError('dataset must be of type DataFrame or Dataset. instead got: '
                                       f'{type(obj).__name__}')

    @classmethod
    def validate_dataset(cls, obj) -> 'Dataset':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Args:
            obj: object to validate as dataset
            check_name (str): check name to print in error

        Returns:
            (Dataset): object that is deepchecks dataset
        """
        if not isinstance(obj, Dataset):
            raise DeepchecksValueError('Check requires dataset to be of type Dataset. instead got: '
                                       f'{type(obj).__name__}')
        if len(obj.data) == 0:
            raise DeepchecksValueError('Check requires a non-empty dataset')

        return obj


def ensure_dataframe_type(obj: t.Any) -> pd.DataFrame:
    """Ensure that given object is of type DataFrame or Dataset and return it as DataFrame. else raise error.

    Args:
        obj: Object to ensure it is DataFrame or Dataset

    Returns:
        (pd.DataFrame)
    """
    if isinstance(obj, pd.DataFrame):
        return obj
    elif isinstance(obj, Dataset):
        return obj.data
    else:
        raise DeepchecksValueError(f'dataset must be of type DataFrame or Dataset, but got: {type(obj).__name__}')
