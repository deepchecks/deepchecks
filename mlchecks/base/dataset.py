from typing import List

import pandas as pd
from pandas_profiling import ProfileReport

__all__ = ['Dataset']


class Dataset(pd.DataFrame):
    def __init__(self,
                 df: pd.DataFrame,
                 features: List[str] = None, cat_features: List[str] = None,
                 label: str = None, use_index: bool = False, index: str = None, date: str = None,
                 *args, **kwargs):

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
        # TODO: add infer logic here
        return []

    def index_col(self) -> pd.Series:
        if self.use_index is True:
            return pd.Series(self.index)
        elif self._index_name is not None:
            return self[self._index_name]
        else:  # No meaningful index to use: Index column not configured, and use_column is False
            return None

    def date_col(self) -> pd.Series:
        if self._date_name:
            return self[self._date_name]
        else:  # Date column not configured in Dataset
            return None

    def cat_features(self) -> List[str]:
        return self._cat_features

    def features(self) -> List[str]:
        return self._features

    def _get_profile(self):
        profile = ProfileReport(self, title="Dataset Report", explorative=True, minimal=True)
        return profile

    def _repr_mimebundle_(self, include, exclude):
        return {'text/html': self._get_profile().to_notebook_iframe()}
