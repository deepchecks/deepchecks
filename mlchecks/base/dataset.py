"""The Dataset module containing the dataset Class and its functions."""
from typing import Union, List
import pandas as pd
from pandas_profiling import ProfileReport

__all__ = ['Dataset', 'validate_dataset', 'validate_column']

from mlchecks.utils import MLChecksValueError


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

        if features:
            self._features = features
        else:
            self._features = [x for x in df.columns if x not in {label, index, date}]

        self._label = label
        self._use_index = use_index
        self._index_name = index
        self._date_name = date

        if cat_features:
            self._cat_features = cat_features
        else:
            self._cat_features = self.infer_categorical_features()

        self._profile = ProfileReport(self, title='Dataset Report', explorative=True, minimal=True)

    def infer_categorical_features(self) -> List[str]:
        """Infers which features are categorical by checking types and number of unique values.

        Returns:
           Out of the list of feature names, returns list of categorical features
        """
        # TODO: add infer logic here
        return []

    def index_col(self) -> Union[pd.Series, None]:
        """Return index column. Index can be a named column or DataFrame index.

        Returns:
           If date column exists, returns a pandas Series of the index column.
        """
        if self.use_index is True:
            return pd.Series(self.index)
        elif self._index_name is not None:
            return self[self._index_name]
        else:  # No meaningful index to use: Index column not configured, and use_column is False
            return None

    def date_col(self) -> Union[pd.Series, None]:
        """If date column exists, returns a pandas Series of the index column.

        Returns:
           (Series): Series of the index column
        """
        if self._date_name:
            return self[self._date_name]
        else:  # Date column not configured in Dataset
            return None

    def label_col(self) -> Union[pd.Series, None]:
        """Return label column.

        Returns:
           Label column
        """
        if self._label:
            return self[self._label]
        else:
            return None

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


def validate_dataset(ds) -> Dataset:
    """Throws error if object is not pandas DataFrame or MLChecks Dataset and returns the object as MLChecks Dataset.

    Args:
        ds: object to validate as dataset

    Returns:
        (Dataset): object converted to MLChecks dataset
    """
    if isinstance(ds, Dataset):
        return ds
    elif isinstance(ds, pd.DataFrame):
        return Dataset(ds)
    else:
        raise MLChecksValueError(f'dataset must be of type DataFrame or Dataset instead got: '
                                 f'{type(ds).__name__}')


def validate_column(column: str, dataset: Dataset) -> List[str]:
    """Validate given column on dataset.

    If column is not None, make sure it exists in the datasets, and return list column name.
    If column is None return list of all columns in the dataset.

    Args:
        column ([None, str]): column name or None
        dataset (Dataset): Dataset working on

    Returns:
        (List[str]): List with column names to work on
    """
    if column is None:
        # If column is None works on all columns
        return [dataset.columns]
    else:
        if not isinstance(column, str):
            raise MLChecksValueError(f"column type must be 'None' or 'str' but got: {type(column).__name__}")
        if len(column) == 0:
            raise MLChecksValueError("column can't be empty string")
        if column not in dataset.columns:
            raise MLChecksValueError(f"column {column} isn't found in the dataset")
        return [column]
