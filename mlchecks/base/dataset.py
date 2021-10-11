from typing import List, Union

import pandas as pd
from pandas_profiling import ProfileReport

__all__ = ['Dataset']


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
                 df: pd.DataFrame,
                 *args,
                 features: List[str] = None, cat_features: List[str] = None,
                 label: str = None, use_index: bool = False, index: str = None, date: str = None,
                 **kwargs):
        """Initiates the Dataset using a pandas DataFrame and Metadata

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
            raise ValueError('parameter use_index cannot be True if index is given')

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

    def infer_categorical_features(self) -> List[str]:
        """Infers which features are categorical by checking types and number of unique values

        Returns:
           Out of the list of feature names, returns list of categorical features

        """
        # TODO: add infer logic here
        return []

    def index_col(self) -> Union[pd.Series, None]:
        """Returns index column. Index can be a named column or DataFrame index.

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
        """
        Returns:
           If date column exists, returns a pandas Series of the index column.
        """

        if self._date_name:
            return self[self._date_name]
        else:  # Date column not configured in Dataset
            return None

    def cat_features(self) -> List[str]:
        """
        Returns:
           List of categorical feature names.
        """
        return self._cat_features

    def features(self) -> List[str]:
        """
        Returns:
           List of feature names.
        """
        return self._features

    def _get_profile(self):
        profile = ProfileReport(self, title='Dataset Report', explorative=True, minimal=True)
        return profile

    def _repr_mimebundle_(self, include, exclude): # pylint: disable=unused-argument TODO: use arguments
        return {'text/html': self._get_profile().to_notebook_iframe()}
