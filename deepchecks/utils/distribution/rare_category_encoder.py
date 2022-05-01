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
"""Module of RareCategoryEncoder."""
from collections import defaultdict
from typing import List, Optional

import pandas as pd

from deepchecks.utils.typing import Hashable

__all__ = ['RareCategoryEncoder']


class RareCategoryEncoder:
    """Encodes rare categories into an "other" parameter.

    Note that this encoder assumes data is received as a DataFrame.

    Parameters
    ----------
    max_num_categories : int , default: 10
        Indicates the maximum number of unique categories in a single categorical column
        (rare categories will be changed to a form of "other")
    cols : Optional[List[Hashable]] , default: None
        Columns to limit the encoder to work on. If non are given will work on all columns given in `fit`
    """

    DEFAULT_OTHER_VALUE = 'OTHER_RARE_CATEGORY'

    def __init__(
        self,
        max_num_categories: int = 10,
        cols: Optional[List[Hashable]] = None
    ):
        self.max_num_categories = max_num_categories
        self.cols = cols
        self._col_mapping = None

    def fit(self, data: pd.DataFrame, y=None):  # noqa # pylint: disable=unused-argument
        """Fit the encoder using given dataframe.

        Parameters
        ----------
        data : pd.DataFrame
            data to fit from
        y :
            Unused, but needed for sklearn pipeline
        """
        self._col_mapping = {}

        if self.cols is not None:
            for col in self.cols:
                self._col_mapping[col] = self._fit_for_series(data[col])
        else:
            for col in data.columns:
                self._col_mapping[col] = self._fit_for_series(data[col])

    def transform(self, data: pd.DataFrame):
        """Transform given data according to columns processed in `fit`.

        Parameters
        ----------
        data : pd.DataFrame
            data to transform
        Returns
        -------
        DataFrame
            transformed data
        """
        if self._col_mapping is None:
            raise RuntimeError('Cannot transform without fitting first')

        if self.cols is not None:
            data = data.copy()
            data[self.cols] = data[self.cols].apply(lambda s: s.map(self._col_mapping[s.name]))
        else:
            data = data.apply(lambda s: s.map(self._col_mapping[s.name]))

        return data

    def fit_transform(self, data: pd.DataFrame, y=None):  # noqa # pylint: disable=unused-argument
        """Run `fit` and `transform` on given data.

        Parameters
        ----------
        data : pd.DataFrame
            data to fit on and transform
        y :
            Unused, but needed for sklearn pipeline
        Returns
        -------
        DataFrame
            transformed data
        """
        self.fit(data)
        return self.transform(data)

    def _fit_for_series(self, series: pd.Series):
        top_values = list(series.value_counts().head(self.max_num_categories).index)
        other_value = self._get_unique_other_value(series)
        mapper = defaultdict(lambda: other_value, {k: k for k in top_values})
        return mapper

    def _get_unique_other_value(self, series: pd.Series):
        unique_values = list(series.unique())
        other = self.DEFAULT_OTHER_VALUE
        i = 0
        while other in unique_values:
            other = self.DEFAULT_OTHER_VALUE + str(i)
            i += 1
        return other
