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
# pylint: skip-file
"""The missing values label correlation check module."""
import typing as t

import numpy as np
import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.correlation_methods import correlation_ratio, theil_u_correlation

__all__ = ['MissingValuesLabelCorrelation']


MLC = t.TypeVar('MLC', bound='MissingValuesLabelCorrelation')


def _is_column_in_df(column, df):
    try:
        return column in df.columns
    except:
        return False


class MissingValuesLabelCorrelation(SingleDatasetCheck):
    """Return the correlation score of missing values with the label.

    If missing values are correlated to the label it points to a systematic bias
    in the data collection, often related to human input. An example could be
     sales data where some fields were only filled if the offer was accepted.
    Parameters
    ----------
    n_top_features: int, default: 5
        Number of features to show, sorted by correlation score
    empty_string_is_na: bool, default: False
        Count empty strings as missing values
    """

    def __init__(
        self,
        n_top_features: int = 5,
        empty_string_is_na: bool = True,
        ** kwargs
    ):
        super().__init__(**kwargs)
        self.n_top_features = n_top_features
        self.empty_string_is_na = empty_string_is_na

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value is a dictionary with correlation of missing values per column to the label.
            data is a bar graph of the correlation of each feature with the label.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset instance with a label.
        """
        dataset = context.get_data_by_kind(dataset_kind)
        dataset.assert_features()
        dataset.assert_label()
        relevant_columns = dataset.features + [dataset.label_name]

        df_corr = self._calculate_correlation_missing_values(
            df=dataset.data[relevant_columns], y=dataset.label_name, task_type=dataset.label_type)

        if context.with_display:
            # TODO fill in display code
            display = None
            # top_to_show = df_corr.head(self.n_top_features)

        else:
            display = None

        return CheckResult(value=df_corr.to_dict(), display=display, header='Missing Values Label Correlation')

    def _calculate_correlation_missing_values(self, df: pd.DataFrame, y: str, task_type: TaskType) -> pd.DataFrame:
        """
        Calculate the correlation of missing values with the target.

        Parameters
        ----------
        df : pandas.DataFrame
            The dataframe that contains the data
        y : pandas.Series
            Target values
        task_type : TaskType
                    Type of problem

        Returns
        -------
        pandas.DataFrame
            Returns a dataframe with all correlations.
        """
        if not isinstance(df, pd.DataFrame):
            raise TypeError(
                f'The dataframe should be a pandas.DataFrame but you passed a {type(df)}\nPlease convert your input to a pandas.DataFrame'
            )
        if not _is_column_in_df(y, df):
            raise ValueError(
                "The 'y' argument should be the name of a dataframe column but the variable that you passed is not a column in the given dataframe.\nPlease review the column name or your dataframe"
            )
        if len(df[[y]].columns) >= 2:
            raise AssertionError(
                f'The dataframe has {len(df[[y]].columns)} columns with the same column name {y}\nPlease adjust the dataframe and make sure that only 1 column has the name {y}'
            )

        num_missing_labels = df[y].isna().sum()
        if num_missing_labels != 0:
            raise ValueError(f'Expected no missing labels but found {num_missing_labels} in {y}')

        if self.empty_string_is_na:
            df = df.copy()
            columns = df.select_dtypes('object').columns
            df[columns] = df[columns].apply(lambda col: col.str.strip().replace("", np.NaN))

        cols = [column for column in df.columns if column != y]
        scores = [self._score(df[column].isna(), df[y], task_type) for column in cols]

        return pd.Series(data=scores, index=cols)

    def _score(self, is_missing: pd.Series, target: pd.Series, task_type: TaskType) -> float:

        if task_type in [TaskType.MULTICLASS, TaskType.BINARY]:
            # asymmetric categorical to categorical corr
            return theil_u_correlation(x=target, y=is_missing)
        elif task_type == TaskType.REGRESSION:
            # categorical to numberical corr
            return correlation_ratio(categorical_data=is_missing, numerical_data=target)
        else:
            raise DeepchecksValueError(f'No correlation method defined yet for TaskType {task_type}')
