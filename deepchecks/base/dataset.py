"""The Dataset module containing the dataset Class and its functions."""
import logging
from typing import Dict, Union, List, Any
import pandas as pd
from pandas.core.dtypes.common import is_float_dtype

from deepchecks.utils.dataframes import filter_columns_with_validation
from deepchecks.errors import DeepchecksValueError
from deepchecks.utils.strings import is_string_column


__all__ = ['Dataset', 'ensure_dataframe_type']


logger = logging.getLogger('deepchecks.dataset')


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
        cat_features: List of names for the categorical features in the DataFrame.
    """

    _features: List[str]
    _label: Union[str, None]
    _use_index: bool
    _index_name: Union[str, None]
    _date_name: Union[str, None]
    cat_features: List[str]
    _data: pd.DataFrame
    _max_categorical_ratio: float
    _max_categories: int

    def __init__(self, df: pd.DataFrame,
                 features: List[str] = None, cat_features: List[str] = None, label: str = None, use_index: bool = False,
                 index: str = None, date: str = None, date_unit_type: str = None, convert_date_: bool = True,
                 max_categorical_ratio: float = 0.01, max_categories: int = 30, max_float_categories: int = 5):
        """Initiate the Dataset using a pandas DataFrame and Metadata.

        Args:
          df: A pandas DataFrame containing data relevant for the training or validating of a ML models
          features: List of names for the feature columns in the DataFrame.
          cat_features: List of names for the categorical features in the DataFrame. In order to disable categorical
                        features inference, pass cat_features=[]
          label: Name of the label column in the DataFrame.
          use_index: Whether to use the dataframe index as the index column, for index related checks.
          index: Name of the index column in the DataFrame.
          date: Name of the date column in the DataFrame.
          date_unit_type: Unit used for conversion if date column is of type int or float.
                          The valid values are 'D', 'h', 'm', 's', 'ms', 'us', and 'ns'.
                          e.g. 's' for seconds, 'ns' for nanoseconds. See pandas.Timestamp unit arg for more detail.
          max_categorical_ratio: The max ratio of unique values in a column in order for it to be inferred as a
                                 categorical feature.
          max_categories: The maximum number of categories in a column in order for it to be inferred as a categorical
                          feature.
          max_float_categories: The maximum number of categories in a float column in order fo it to be inferred as a
                                categorical feature

        """
        self._data = df.copy()

        # Validations
        if use_index is True and index is not None:
            raise DeepchecksValueError('parameter use_index cannot be True if index is given')
        if index is not None and index not in self._data.columns:
            error_message = f'index column {index} not found in dataset columns.'
            if index == 'index':
                error_message += ' If you attempted to use the dataframe index, set use_index to True instead.'
            raise DeepchecksValueError(error_message)
        if date is not None and date not in self._data.columns:
            raise DeepchecksValueError(f'date column {date} not found in dataset columns')
        if label is not None and label not in self._data.columns:
            raise DeepchecksValueError(f'label column {label} not found in dataset columns')

        if features:
            if any(x not in self._data.columns for x in features):
                raise DeepchecksValueError(f'Features must be names of columns in dataframe.'
                                           f' Features {set(features) - set(self._data.columns)} have not been '
                                           f'found in input dataframe.')
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
        self._max_float_categories = max_float_categories

        if self._label_name in self.features():
            raise DeepchecksValueError(f'label column {self._label_name} can not be a feature column')

        if self._label_name:
            try:
                self.check_compatible_labels()
            except DeepchecksValueError as e:
                logger.warning(str(e))

        if self._date_name in self.features():
            raise DeepchecksValueError(f'date column {self._date_name} can not be a feature column')

        if self._index_name in self.features():
            raise DeepchecksValueError(f'index column {self._index_name} can not be a feature column')

        if cat_features is not None:
            if set(cat_features).intersection(set(self._features)) != set(cat_features):
                raise DeepchecksValueError(f'Categorical features must be a subset of features. '
                                           f'Categorical features {set(cat_features) - set(self._features)} '
                                           f'have not been found in feature list.')
            self.cat_features = cat_features
        else:
            self.cat_features = self.infer_categorical_features()

        if self._date_name and convert_date_:
            self._data[self._date_name] = self._data[self._date_name].apply(pd.Timestamp, unit=date_unit_type)

    @property
    def data(self) -> pd.DataFrame:
        """Return the data of dataset."""
        return self._data

    def copy(self, new_data):
        """Create a copy of this Dataset with new data."""
        # Filter out if columns were dropped
        features = list(set(self._features).intersection(new_data.columns))
        cat_features = list(set(self.cat_features).intersection(new_data.columns))
        label = self._label_name if self._label_name in new_data.columns else None
        index = self._index_name if self._index_name in new_data.columns else None
        date = self._date_name if self._date_name in new_data.columns else None

        return Dataset(new_data, features=features, cat_features=cat_features, label=label, use_index=self._use_index,
                       index=index, date=date, convert_date_=False, max_categorical_ratio=self._max_categorical_ratio,
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

        # Checking for categorical dtypes
        cat_dtypes = self.data.select_dtypes(include='category')
        if len(cat_dtypes.columns) > 0:
            return list(cat_dtypes.columns)

        for col in self._features:
            if self.is_categorical(col):
                cat_columns.append(col)

        if len(cat_columns) > 0:
            if len(cat_columns) < 7:
                print(f'Automatically inferred these columns as categorical features: {", ".join(cat_columns)}. \n')
            else:
                print(f'Some columns have been inferred as categorical features: {", ".join(cat_columns[:7])}. \n '
                      f'and more... \n'
                      f'For the full list of columns, use dataset.cat_features')

        return cat_columns

    def is_categorical(self, col_name: str) -> bool:
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

    def features(self) -> List[str]:
        """Return list of feature names.

        Returns:
           List of feature names.
        """
        return self._features

    def features_columns(self) -> Union[pd.DataFrame, None]:
        """Return features columns if exists.

        Returns:
           Features columns
        """
        return self.data[self._features] if self._features else None

    def show_columns_info(self) -> Dict:
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
        if is_string_column(self.label_col()):
            raise DeepchecksValueError('String labels are not supported')
        elif pd.isnull(self.label_col()).any():
            raise DeepchecksValueError('Can\'t have null values in label column')

    # Validations:

    def validate_label(self, check_name: str):
        """
        Throws error if dataset does not have a label.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have a label

        """
        if self.label_name() is None:
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have a label column')
        self.check_compatible_labels()

    def validate_date(self, check_name: str):
        """
        Throws error if dataset does not have a date column.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have a date column

        """
        if self.date_name() is None:
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have a date column')

    def validate_index(self, check_name: str):
        """
        Throws error if dataset does not have an index column / does not use dataframe index as index.

        Args:
            check_name (str): check name to print in error

        Raises:
            DeepchecksValueError if dataset does not have an index

        """
        if self.index_name() is None:
            raise DeepchecksValueError(f'Check {check_name} requires dataset to have an index column')

    def filter_columns_with_validation(self, columns: Union[str, List[str], None] = None,
                                       ignore_columns: Union[str, List[str], None] = None) -> 'Dataset':
        """Filter dataset columns by given params.

        Args:
            columns (Union[str, List[str], None]): Column names to keep.
            ignore_columns (Union[str, List[str], None]): Column names to drop.
        Raise:
            DeepchecksValueError: In case one of columns given don't exists raise error
        """
        new_data = filter_columns_with_validation(self.data, columns, ignore_columns)
        if new_data.equals(self.data):
            return self
        else:
            return self.copy(new_data)

    def validate_shared_features(self, other, check_name: str) -> List[str]:
        """
        Return the list of shared features if both datasets have the same feature column names. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            check_name (str): check name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            DeepchecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, check_name)
        if sorted(self.features()) == sorted(other.features()):
            return self.features()
        else:
            raise DeepchecksValueError(f'Check {check_name} requires datasets to share the same features')

    def validate_shared_categorical_features(self, other, check_name: str) -> List[str]:
        """
        Return list of categorical features if both datasets have the same categorical features. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            check_name (str): check name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            DeepchecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, check_name)
        if sorted(self.cat_features) == sorted(other.cat_features):
            return self.cat_features
        else:
            raise DeepchecksValueError(f'Check {check_name} requires datasets to share'
                                       f' the same categorical features. Possible reason is that some columns were'
                                       f'inferred incorrectly as categorical features. To fix this, manually edit the '
                                       f'categorical features using Dataset(cat_features=<list_of_features>')

    def validate_shared_label(self, other, check_name: str) -> str:
        """
        Return the list of shared features if both datasets have the same feature column names. Else, raise error.

        Args:
            other: Expected to be Dataset type. dataset to compare features list
            check_name (str): check name to print in error

        Returns:
            List[str] - list of shared features names

        Raises:
            DeepchecksValueError if datasets don't have the same features

        """
        Dataset.validate_dataset(other, check_name)
        if self.label_name() == other.label_name():
            return self.label_name()
        else:
            raise DeepchecksValueError(f'Check {check_name} requires datasets to share the same label')

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
            raise DeepchecksValueError(f'dataset must be of type DataFrame or Dataset. instead got: '
                                       f'{type(obj).__name__}')

    def validate_model(self, model):
        """Check model is able to predict on the dataset.

        Raise:
            DeepchecksValueError: if dataset does not match model
        """
        try:
            model.predict(self.features_columns().head(1))
        except Exception as exc:
            raise DeepchecksValueError('Got error when trying to predict with model on dataset') from exc

    @classmethod
    def validate_dataset(cls, obj, check_name: str) -> 'Dataset':
        """Throws error if object is not deepchecks Dataset and returns the object if deepchecks Dataset.

        Args:
            obj: object to validate as dataset
            check_name (str): check name to print in error

        Returns:
            (Dataset): object that is deepchecks dataset
        """
        if not isinstance(obj, Dataset):
            raise DeepchecksValueError(f'Check {check_name} requires dataset to be of type Dataset. instead got: '
                                       f'{type(obj).__name__}')
        if len(obj.data) == 0:
            raise DeepchecksValueError(f'Check {check_name} required a non-empty dataset')

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
        raise DeepchecksValueError(f'dataset must be of type DataFrame or Dataset, but got: {type(obj).__name__}')
