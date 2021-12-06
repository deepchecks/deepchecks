# Deepchecks
# Copyright (C) 2021 Deepchecks
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
"""The dataset_info check module."""
from typing import Union
import pandas as pd
from pandas_profiling import ProfileReport

from deepchecks import CheckResult, Dataset, SingleDatasetBaseCheck, ensure_dataframe_type

__all__ = ['DatasetInfo']


class DatasetInfo(SingleDatasetBaseCheck):
    """Summarize given dataset information based on pandas_profiling package."""

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Arguments:
            dataset: Dataset - The dataset object
            model: any = None - not used in the check

        Returns:
            the output of the dataset_info check
        """
        return self._dataset_info(dataset)

    def _dataset_info(self, dataset: Union[Dataset, pd.DataFrame]):
        """Run check.

        Args:
           dataset (Dataset): A dataset object
        Returns:
           CheckResult: value is tuple that represents the shape of the dataset

        Raises:
            DeepchecksValueError: If the object is not of a supported type
        """
        dataset: pd.DataFrame = ensure_dataframe_type(dataset)

        def display():
            profile = ProfileReport(dataset, title='Dataset Report', explorative=True, minimal=True)
            profile.to_notebook_iframe()

        return CheckResult(dataset.shape, check=self.__class__, display=display)
