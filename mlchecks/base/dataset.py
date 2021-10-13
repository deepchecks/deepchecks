"""The Dataset module containing the dataset Class and its functions."""
from typing import Union, List
import pandas as pd
from pandas_profiling import ProfileReport
import warnings
from mlchecks.utils import MLChecksValueError


PANDAS_USER_ATTR_WARNING_STR = "Pandas doesn't allow columns to be created via a new attribute name - "\
                               "see "\
                               "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute"\
                               "-access"


__all__ = ['Dataset', 'validate_dataset_or_dataframe', 'validate_dataset', 'single_column_or_all']


class Dataset(pd.DataFrame):
    """Dataset extends pandas DataFrame to provide ML related metadata.

    The Dataset class is a pandas DataFrame containing additional data and method intended for easily accessing
    metadata relevant for the training or validating of a ML models.

    Attributes:
        _features: List of names for the feature columns in the DataFrame.
        _label: Name of the label column in the DataFrame.
        _use_index: Boolean value controlling whether the DataFrame index will be used by the index_col() method.
        _index_name: Name of the index column in the DataFrame.
        _date_name: Name of the date column in the DataFrame.
        _cat_features: List of names for the categorical features in the DataFrame.
    """

    def __init__(self,
                 df: pd.DataFrame, *args,
                 features: List[str] = None, cat_features: List[str] = None,
                 label: str = None, use_index: bool = False, index: str = None, date: str = None,
                 **kwargs):
        """Initiate the Dataset using a pandas DataFrame and Metadata.

        Args:
          df: A pandas DataFrame containing data relevant for the training or validating of a ML models
          features: List of names for the feature columns in the DataFrame.
          cat_features: List of names for the categorical features in the DataFrame.
          label: Name of the label column in the DataFrame.
          use_index: Name of the index column in the DataFrame.
          index: Name of the index column in the DataFrame.
          date: Name of the date column in the DataFrame.

        """
        super().__init__(df, *args, **kwargs)

        # Validations
        if use_index is True and index is not None:
            raise MLChecksValueError('parameter use_index cannot be True if index is given')
        if date is not None and date not in self.columns:
            raise MLChecksValueError(f'date column {date} not found in dataset columns')
        if label is not None and label not in self.columns:
            raise MLChecksValueError(f'label column {label} not found in dataset columns')

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore",
                                    message=PANDAS_USER_ATTR_WARNING_STR)
            if features:
                self._features = features
            else:
                self._features = [x for x in df.columns if x not in {label, index, date}]

            self._label_name = label
            self._use_index = use_index
            self._index_name = index
            self._date_name = date

            if cat_features:
                self._cat_features = cat_features
            else:
                self._cat_features = self.infer_categorical_features()

            self._profile = ProfileReport(self, title='Dataset Report', explorative=True, minimal=True)

    def n_samples(self):
        """Return number of samples in dataframe.

        Returns:
           Number of samples in dataframe
        """
        return self.shape[0]

    def infer_categorical_features(self) -> List[str]:
        """Infers which features are categorical by checking types and number of unique values.

        Returns:
           Out of the list of feature names, returns list of categorical features
        """
        # TODO: add infer logic here
        return []

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
            return pd.Series(self.index)
        elif self._index_name is not None:
            return self[self._index_name]
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
        return self[self._date_name] if self._date_name is not None else None

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
        return self[self._label_name] if self._label_name is not None else None

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

    def get_profile(self):
        """Return the pandas profiling object including the statistics of the dataset.

        Returns:
            The pandas profiling object including the statistics of the dataset
        """
        return self._profile

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
        validate_dataset(other, function_name)
        if sorted(self.features()) == sorted(other.features()):
            return self.features()
        else:
            raise MLChecksValueError(f'function {function_name} requires datasets to share the same features')

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
        validate_dataset(other, function_name)
        if sorted(self.label_name()) == sorted(other.label_name()):
            return self.label_name()
        else:
            raise MLChecksValueError(f'function {function_name} requires datasets to share the same label')


def validate_dataset_or_dataframe(obj) -> Dataset:
    """Throws error if object is not pandas DataFrame or MLChecks Dataset and returns the object as MLChecks Dataset.

    Args:
        obj: object to validate as dataset

    Returns:
        (Dataset): object converted to MLChecks dataset
    """
    if isinstance(obj, Dataset):
        return obj
    elif isinstance(obj, pd.DataFrame):
        return Dataset(obj)
    else:
        raise MLChecksValueError(f'dataset must be of type DataFrame or Dataset. instead got: '
                                 f'{type(obj).__name__}')


def validate_dataset(obj, function_name: str) -> Dataset:
    """Throws error if object is not MLChecks Dataset and returns the object if MLChecks Dataset.

    Args:
        obj: object to validate as dataset
        function_name (str): function name to print in error

    Returns:
        (Dataset): object that is MLChecks dataset
    """
    if isinstance(obj, Dataset):
        return obj
    else:
        raise MLChecksValueError(f'function {function_name} requires dataset to be of type Dataset. instead got: '
                                 f'{type(obj).__name__}')


def single_column_or_all(dataset: Dataset, column: str = None) -> List[str]:
    """Validate given column on dataset.

    If column is not None, make sure it exists in the datasets, and return list containing only column name.
    If column is None return list of all columns in the dataset.

    Args:
        dataset (Dataset): Dataset working on
        column ([None, str]): column name or None

    Returns:
        (List[str]): List with column names to work on
    """
    if column is None:
        # If column is None works on all columns
        return list(dataset.columns)
    else:
        if not isinstance(column, str):
            raise MLChecksValueError(f"column type must be 'None' or 'str' but got: {type(column).__name__}")
        if len(column) == 0:
            raise MLChecksValueError("column can't be empty string")
        if column not in dataset.columns:
            raise MLChecksValueError(f"column {column} isn't found in the dataset")
        return [column]
