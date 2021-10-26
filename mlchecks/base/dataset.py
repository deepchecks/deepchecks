"""The Dataset module containing the dataset Class and its functions."""
from typing import Union, List, Any
import pandas as pd

from mlchecks.base.dataframe_utils import filter_columns_with_validation
from mlchecks.utils import MLChecksValueError


__all__ = ['Dataset', 'ensure_dataframe_type']


class Dataset:
    """Dataset wraps pandas DataFrame together with ML related metadata.

    The Dataset class is containing additional data and methods intended for easily accessing
    metadata relevant for the training or validating of a ML models.

    Attributes:
        _features: List of names for the feature columns in the DataFrame.
        _label: Name of the label column in the DataFrame.
        _use_index: Boolean value controlling whether the DataFrame index will be used by the index_col() method.
        _index_name: Name of the index column in the DataFrame.
        _date_name: Name of the date column in the DataFrame.
        _cat_features: List of names for the categorical features in the DataFrame.
    """

    _features: List[str]
    _label: Union[str, None]
    _use_index: bool
    _index_name: Union[str, None]
    _date_name: Union[str, None]
    _cat_features: List[str]
    _data: pd.DataFrame
    _max_categorical_ratio: float
    _max_categories: int

    def __init__(self, df: pd.DataFrame,
                 features: List[str] = None, cat_features: List[str] = None, label: str = None, use_index: bool = False,
                 index: str = None, date: str = None, date_unit_type: str = None, _convert_date: bool = True,
                 max_categorical_ratio: float = 0.001, max_categories: int = 100):
        """Initiate the Dataset using a pandas DataFrame and Metadata.

        Args:
          df: A pandas DataFrame containing data relevant for the training or validating of a ML models
          features: List of names for the feature columns in the DataFrame.
          cat_features: List of names for the categorical features in the DataFrame. In order to disable categorical
                        features inference, pass cat_features=[]
          label: Name of the label column in the DataFrame.
          use_index: Name of the index column in the DataFrame.
          index: Name of the index column in the DataFrame.
          date: Name of the date column in the DataFrame.
          date_unit_type: Unit used for conversion if date column is of type int or float.
                          The valid values are 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'.
                          e.g. 's' for seconds, 'ns' for nanoseconds. See pandas.Timestamp unit arg for more detail.
          max_categorical_ratio: The max ratio of unique values in a column in order for it to be inferred as a
                                 categorical feature.
          max_categories: The maximum number of categories in a column in order for it to be inferred as a categorical
                          feature.

        """
        self._data = df.copy()

        # Validations
        if use_index is True and index is not None:
            raise MLChecksValueError('parameter use_index cannot be True if index is given')
        if date is not None and date not in self._data.columns:
            raise MLChecksValueError(f'date column {date} not found in dataset columns')
        if label is not None and label not in self._data.columns:
            raise MLChecksValueError(f'label column {label} not found in dataset columns')

        if features:
            self._features = features
        else:
            self._features = [x for x in self._data.columns if x not in {label, index, date}]

        self._label_name = label
        self._use_index = use_index
        self._index_name = index
        self._date_name = date
        self._date_unit_type = date_unit_type
        self._max_categorical_ratio = max_categorical_ratio
        self._max_categories = max_categories

        if cat_features is not None:
            self._cat_features = cat_features
        else:
            self._cat_features = self.infer_categorical_features()

        if self._date_name and _convert_date:
            self._data[self._date_name] = self._data[self._date_name].apply(pd.Timestamp, unit=date_unit_type)

    @property
    def data(self) -> pd.DataFrame:
        """Return the data of dataset."""
        return self._data

    def copy(self, new_data):
        """Create a copy of this Dataset with new data."""
        # Filter out if columns were dropped
        features = list(set(self._features).intersection(new_data.columns))
        cat_features = list(set(self._cat_features).intersection(new_data.columns))
        label = self._label if self._label in new_data.columns else None
        index = self._index_name if self._index_name in new_data.columns else None
        date = self._date_name if self._date_name in new_data.columns else None

        return Dataset(new_data, features=features, cat_features=cat_features, label=label, use_index=self._use_index,
                       index=index, date=date, _convert_date=False, max_categorical_ratio=self._max_categorical_ratio,
                       max_categories=self._max_categories)

    def n_samples(self):
        """Return number of samples in dataframe.

        Returns:
           Number of samples in dataframe
        """
        return self.data.shape[0]

    def infer_categorical_features(self) -> List[str]:
        """Infers which features are categorical by checking types and number of unique values.

        Returns:
           Out of the list of feature names, returns list of categorical features
        """
        cat_columns = []

        for col in self.data.columns:
            num_unique = self.data[col].nunique(dropna=True)
            if num_unique / len(self.data[col].dropna()) < self._max_categorical_ratio\
                    or num_unique <= self._max_categories:
                cat_columns.append(col)

        return cat_columns

    def index_name(self) -> Union[str, None]:
        """If index column exists, return its name.

        Returns:
           (str) index column name
        """
        return self._index_name

    def index_col(self) -> Union[pd.Series, None]:
        """Return index column. Index can be a named column or DataFrame index.

        Returns:
           If date column exists, returns a pandas Series of the index column.
        """
        if self._use_index is True:
            return pd.Series(self.data.index)
        elif self._index_name is not None:
            return self.data[self._index_name]
        else:  # No meaningful index to use: Index column not configured, and use_column is False
            return None

    def date_name(self) -> Union[str, None]:
        """If date column exists, return its name.

        Returns:
           (str) date column name
        """
        return self._date_name

    def date_col(self) -> Union[pd.Series, None]:
        """Return date column if exists.

        Returns:
           (Series): Series of the date column
        """
        return self.data[self._date_name] if self._date_name else None

    def label_name(self) -> Union[str, None]:
        """If label column exists, return its name.

        Returns:
           (str) Label name
        """
        return self._label_name

    def label_col(self) -> Union[pd.Series, None]:
        """Return label column if exists.

        Returns:
           Label column
        """
        return self.data[self._label_name] if self._label_name else None

    def cat_features(self) -> List[str]:
        """Return List of categorical feature names.

        Returns:
           List of categorical feature names.
        """
        return self._cat_features

    def features(self) -> List[str]:
        """Return list of feature names.

        Returns:
           List of feature names.
        """
        return self._features

    # Validations:

    def validate_label(self, function_name: str):
        """
        Throws error if dataset does not have a label.

        Args:
            function_name (str): function name to print in error

        Raises:
            MLChecksValueError if dataset does not have a label

        """
        if self.label_name() is None:
            raise MLChecksValueError(f'function {function_name} requires dataset to have a label column')

    def validate_date(self, function_name: str):
        """
        Throws error if dataset does not have a date column.

        Args:
            function_name (str): function name to print in error

        Raises:
            MLChecksValueError if dataset does not have a date column

        """
        if self.date_name() is None:
            raise MLChecksValueError(f'function {function_name} requires dataset to have a date column')

    def validate_index(self, function_name: str):
        """
        Throws error if dataset does not have an index column / does not use dataframe index as index.

        Args:
            function_name (str): function name to print in error

        Raises:
            MLChecksValueError if dataset does not have an index

        """
        if self.index_name() is None:
            raise MLChecksValueError(f'function {function_name} requires dataset to have an index column')

    def filter_columns_with_validation(self, columns: Union[str, List[str], None] = None,
                                       ignore_columns: Union[str, List[str], None] = None) -> 'Dataset':
        """Filter dataset columns by given params.

        Args:
            columns (Union[str, List[str], None]): Column names to keep.
            ignore_columns (Union[str, List[str], None]): Column names to drop.
        Raise:
            MLChecksValueError: In case one of columns given don't exists raise error
        """
        new_data = filter_columns_with_validation(self.data, columns, ignore_columns)
        if new_data == self.data:
            return self
        else:
            return self.copy(new_data)

    def validate_shared_features(self, other, function_name: str) -> List[str]:
        """
        Return the list of shared features if both datasets have the same feature column names. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            function_name (str): function name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            MLChecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, function_name)
        if sorted(self.features()) == sorted(other.features()):
            return self.features()
        else:
            raise MLChecksValueError(f'function {function_name} requires datasets to share the same features')

    def validate_shared_categorical_features(self, other, function_name: str) -> List[str]:
        """
        Return the list of shared categorical features if both datasets have the same categorical feature column names.
        Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            function_name (str): function name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            MLChecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, function_name)
        if sorted(self.cat_features()) == sorted(other.cat_features()):
            return self.cat_features()
        else:
            raise MLChecksValueError(f'function {function_name} requires datasets to share'
                                     f' the same categorical features')

    def validate_shared_label(self, other, function_name: str) -> str:
        """
        Return the list of shared features if both datasets have the same feature column names. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            function_name (str): function name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            MLChecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, function_name)
        if sorted(self.label_name()) == sorted(other.label_name()):
            return self.label_name()
        else:
            raise MLChecksValueError(f'function {function_name} requires datasets to share the same label')

    @classmethod
    def validate_dataset_or_dataframe(cls, obj) -> 'Dataset':
        """
        Raise error if object is not pandas DataFrame or MLChecks Dataset and returns the object as MLChecks Dataset.

        Args:
            obj: object to validate as dataset

        Returns:
            (Dataset): object converted to MLChecks dataset
        """
        if isinstance(obj, Dataset):
            if len(obj.data) == 0:
                raise MLChecksValueError('dataset cannot be empty')
            return obj
        elif isinstance(obj, pd.DataFrame):
            if len(obj) == 0:
                raise MLChecksValueError('dataset cannot be empty')
            return Dataset(obj)
        else:
            raise MLChecksValueError(f'dataset must be of type DataFrame or Dataset. instead got: '
                                     f'{type(obj).__name__}')

    @classmethod
    def validate_dataset(cls, obj, function_name: str) -> 'Dataset':
        """Throws error if object is not MLChecks Dataset and returns the object if MLChecks Dataset.

        Args:
            obj: object to validate as dataset
            function_name (str): function name to print in error

        Returns:
            (Dataset): object that is MLChecks dataset
        """
        if not isinstance(obj, Dataset):
            raise MLChecksValueError(f'function {function_name} requires dataset to be of type Dataset. instead got: '
                                     f'{type(obj).__name__}')
        if len(obj.data) == 0:
            raise MLChecksValueError(f'function {function_name} required a non-empty dataset')

        return obj


def ensure_dataframe_type(obj: Any) -> pd.DataFrame:
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
        raise MLChecksValueError(f'dataset must be of type DataFrame or Dataset, but got: {type(obj).__name__}')
