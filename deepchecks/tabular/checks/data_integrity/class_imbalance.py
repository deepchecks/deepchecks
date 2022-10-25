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
"""Module contains class_imbalance check."""
import typing as t

import plotly.express as px

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.tabular import Context, SingleDatasetCheck
from deepchecks.utils.strings import format_number
from deepchecks.utils.typing import Hashable

__all__ = ['ClassImbalance']


class ClassImbalance(SingleDatasetCheck):
    """Check if a dataset is imbalanced by looking at the target variable distribution.

    Parameters
    ----------
    n_top_labels: int, default: 5
        Number of labels to show in display graph
    ignore_nan: bool, default True
        Whether to ignore NaN values in the target variable when counting
        the number of unique values.
    """

    def __init__(
            self,
            n_top_labels: int = 5,
            ignore_nan: bool = True,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_top_labels = n_top_labels
        self.ignore_nan = ignore_nan

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run the check.

        Returns
        -------
        CheckResult
            value of result is a dict of all unique labels with number of unique values
            in format {label: number_of_uniques} display is a series with labels
            and their normalized count
        """
        dataset = context.get_data_by_kind(dataset_kind)
        context.assert_classification_task()
        label = dataset.label_col

        vc_ser = label.value_counts(normalize=True, dropna=self.ignore_nan)
        vc_ser = vc_ser.round(2)

        if context.with_display:
            vc_ser_plot = vc_ser.head(self.n_top_labels).copy()
            xaxis_layout = dict(
                title='Class',
                type='category',
                # NOTE:
                # the range, in this case, is needed to fix a problem with
                # too wide bars when there are only one or two of them in
                # the plot, plus it also centralizes them in the plot.
                # The min value of the range (range(min, max)) is bigger because
                # otherwise bars will not be centralized on the plot, they will
                # appear on the left part of the plot (that is probably because of zero)
                range=(-3, len(vc_ser.index) + 2)
            )
            fig = px.bar(vc_ser_plot, x=vc_ser_plot.index, y=vc_ser_plot.values,
                         text=vc_ser_plot.values.astype(str),
                         title='Class Label Distribution').update_layout(
                yaxis_title='Frequency', height=400,
                xaxis=xaxis_layout)
            fig.update_traces(textposition='outside')
            fig.update_layout(yaxis_range=[0, 1])
            if self.n_top_labels < len(vc_ser):
                text = f'* showing only the top {self.n_top_labels} labels, you can change it ' \
                       f'by using n_top_labels param'
            else:
                text = ''
            display = [fig, text]
        else:
            display = None

        return CheckResult(vc_ser.to_dict(), display=display)

    def add_condition_class_ratio_less_than(self, class_imbalance_ratio_th: float = 0.1):
        """Add condition - ratio between least to most frequent labels.

        This ratio is compared to class_imbalance_ratio_th.

        Parameters
        ----------
        class_imbalance_ratio_th: float, default: 0.1
            threshold for least frequent label to most frequent label.
        """
        name = 'The ratio between least frequent label to most frequent label ' \
               f'is less than or equal {class_imbalance_ratio_th}'

        def threshold_condition(result: t.Dict[Hashable, float]) -> ConditionResult:
            class_ratio = result[list(result.keys())[-1]] / result[list(result.keys())[0]]
            details = f'The ratio between least to most frequent label is {format_number(class_ratio)}'
            if class_ratio >= class_imbalance_ratio_th:
                return ConditionResult(ConditionCategory.WARN, details)

            return ConditionResult(ConditionCategory.PASS, details)

        return self.add_condition(name, threshold_condition)
