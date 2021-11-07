"""module contains Identifier Leakage check."""
from typing import Union

import pandas as pd

from mlchecks import Dataset
from mlchecks.base.check import CheckResult, SingleDatasetBaseCheck
from mlchecks.plot_utils import create_colorbar_barchart_for_check
from mlchecks.utils import MLChecksValueError
import ppscore as pps


__all__ = ['IdentifierLeakage']


class IdentifierLeakage(SingleDatasetBaseCheck):
    """Search for leakage in identifiers (Date, Index)."""
    def __init__(self, ppscore_params=None, **params):
        """Initialize the IdentifierLeakage check.

        Args:
            ppscore_params: dictionary containing params to pass to ppscore predictor
        """
        super().__init__(**params)
        self.ppscore_params = ppscore_params

    def run(self, dataset: Dataset, model=None) -> CheckResult:
        """Check if identifiers (Index/Date) can be used to predict the label.

        Args:
          dataset(Dataset): any dataset.
          model: ignored in check (default: None).

        Returns:
            (CheckResult):
                value is a dictionary with PPS per feature column.
                data is a bar graph of the PPS of each feature.

        Raises:
            MLChecksValueError: If the object is not a Dataset instance with a label
        """
        return self._identifier_leakage(dataset, **self.params)

    def _identifier_leakage(self, dataset: Union[pd.DataFrame, Dataset], ppscore_params=None) -> CheckResult:
        Dataset.validate_dataset(dataset, self._identifier_leakage.__name__)
        dataset.validate_label(self._identifier_leakage.__name__)
        ppscore_params = ppscore_params or {}

        relevant_columns = list(filter(None, [dataset.date_name(), dataset.index_name(), dataset.label_name()]))

        if len(relevant_columns) == 1:
            raise MLChecksValueError('Dataset needs to have a date or index column.')

        df_pps = pps.predictors(df=dataset.data[relevant_columns], y=dataset.label_name(), random_seed=42,
                                **ppscore_params)
        df_pps = df_pps.set_index('x', drop=True)
        s_ppscore = df_pps['ppscore']

        def plot():
            # Create graph:
            create_colorbar_barchart_for_check(x=s_ppscore.index, y=s_ppscore.values,
                                               ylabel='predictive power score (PPS)',
                                               xlabel='Identifiers', color_map='gist_heat_r', color_shift_midpoint=0.1,
                                               color_label='PPS', check_name=self.__class__.__name__)

        text = ['The PPS represents the ability of a feature to single-handedly predict another feature or label.',
                'For Identifier columns (Index/Date) PPS should be nearly 0, otherwise date and index have some '
                'predictive effect on the label.']

        return CheckResult(value=s_ppscore.to_dict(), display=[plot, *text], check=self.run)
