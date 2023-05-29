# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The base abstract functionality for label drift checks."""
import abc
import typing as t

import pandas as pd
from typing_extensions import Self

from deepchecks import CheckResult, ConditionCategory, ConditionResult
from deepchecks.utils.distribution.drift import calc_drift_and_plot, get_drift_plot_sidenote
from deepchecks.utils.strings import format_number

__all__ = ['LabelDriftAbstract']


class LabelDriftAbstract(abc.ABC):
    """Base class for label drift checks."""

    margin_quantile_filter: float = 0.025
    max_num_categories_for_drift: t.Optional[int]
    min_category_size_ratio: float
    max_num_categories_for_display: t.Optional[int]
    show_categories_by: str
    numerical_drift_method: str = 'KS'
    categorical_drift_method: str
    balance_classes: bool
    ignore_na: bool
    min_samples: int
    n_samples: t.Optional[int]
    random_state: int
    add_condition: t.Callable[..., t.Any]

    def _calculate_label_drift(self, train_column, test_column, label_name: str, column_type: str, with_display: bool,
                               dataset_names: t.Optional[t.Tuple[str, str]]) -> CheckResult:

        drift_score, method, display = calc_drift_and_plot(
            train_column=pd.Series(train_column),
            test_column=pd.Series(test_column),
            value_name=label_name,
            column_type=column_type,
            margin_quantile_filter=self.margin_quantile_filter,
            max_num_categories_for_drift=self.max_num_categories_for_drift,
            min_category_size_ratio=self.min_category_size_ratio,
            max_num_categories_for_display=self.max_num_categories_for_display,
            show_categories_by=self.show_categories_by,
            numerical_drift_method=self.numerical_drift_method,
            categorical_drift_method=self.categorical_drift_method,
            balance_classes=self.balance_classes,
            ignore_na=self.ignore_na,
            min_samples=self.min_samples,
            raise_min_samples_error=True,
            with_display=with_display,
            dataset_names=dataset_names
        )

        values_dict = {'Drift score': drift_score, 'Method': method}

        if with_display:
            displays = ["""<span>
                        The Drift score is a measure for the difference between two distributions, in this check -
                        the test and train distributions.<br> The check shows the drift score
                        and distributions for the label. </span>""",
                        get_drift_plot_sidenote(self.max_num_categories_for_display, self.show_categories_by),
                        display]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Label Drift')

    def add_condition_drift_score_less_than(self, max_allowed_drift_score: float = 0.15) -> Self:
        """
        Add condition - require drift score to be less than the threshold.

        The industry standard for PSI limit is above 0.2.
        There are no common industry standards for other drift methods, such as Cramer's V,
        Kolmogorov-Smirnov and Earth Mover's Distance.

        Parameters
        ----------
        max_allowed_drift_score: float , default: 0.15
            the max threshold for the categorical variable drift score
        Returns
        -------
        ConditionResult
            False if any column has passed the max threshold, True otherwise
        """

        def condition(result: t.Dict) -> ConditionResult:
            drift_score = result['Drift score']
            method = result['Method']

            details = f'Label\'s drift score {method} is {format_number(drift_score)}'
            category = ConditionCategory.FAIL if drift_score > max_allowed_drift_score else ConditionCategory.PASS
            return ConditionResult(category, details)

        return self.add_condition(f'Label drift score < {max_allowed_drift_score}', condition)
