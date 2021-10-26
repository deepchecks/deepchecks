from collections import defaultdict
import pandas as pd
from typing import List

__all__ = ['RareCategoryEncoder']


class RareCategoryEncoder:
    """
    Encodes rare categories into an "other" parameter.
    Note that this encoder assumes data is received as a DataFrame.
    """
    DEFAULT_OTHER_VALUE = 'OTHER_RARE_CATEGORY'

    def __init__(self, max_num_categories: int = 10, cols: List[str]=None):
        self.max_num_categories = max_num_categories
        self.cols = cols
        self._col_mapping = None

    def fit(self, x: pd.DataFrame):
        if self.cols is not None:
            self._col_mapping = x[self.cols].apply(self._fit_for_series, axis=0)
        else:
            self._col_mapping = x.apply(self._fit_for_series, axis=0)

    def transform(self, x):
        if self._col_mapping is None:
            raise RuntimeError('Cannot transform without fitting first')

        if self.cols is not None:
            x = x.copy()
            x[self.cols] = x[self.cols].apply(lambda s: s.map(self._col_mapping[s.name]))
        else:
            x = x.apply(lambda s: s.map(self._col_mapping[s.name]))
        return x

    def fit_transform(self, x: pd.DataFrame):
        self.fit(x)
        return self.transform(x)

    def _fit_for_series(self, x: pd.Series):
        top_values = list(x.value_counts().head(self.max_num_categories).index)
        other_value = self._get_unique_other_value(x)
        mapper = defaultdict(lambda: other_value, {k: k for k in top_values})
        return mapper

    def _get_unique_other_value(self, x: pd.Series):
        unique_values = x.unique()
        other = self.DEFAULT_OTHER_VALUE
        i = 0
        while other in unique_values:
            other = self.DEFAULT_OTHER_VALUE + str(i)
            i += 1
        return other
