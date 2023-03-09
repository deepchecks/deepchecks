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
"""Module containing the base check for single dataset performance checks."""
from abc import ABC
from typing import Dict, List, Optional

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, SingleDatasetBaseCheck
from deepchecks.core.checks import CheckConfig  # pylint: disable=unused-import
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.core.reduce_classes import ReduceMixin
from deepchecks.utils.docref import doclink
from deepchecks.utils.strings import format_number


class BaseSingleDatasetPerformance(SingleDatasetBaseCheck, ReduceMixin, ABC):
    """Base check for checks that summarize given model performance on a dataset based on selected scorers."""

    def config(
        self,
        include_version: bool = True
    ) -> 'CheckConfig':
        """Return check configuration."""
        if isinstance(self.scorers, dict):
            for k, v in self.scorers.items():
                if not isinstance(v, str):
                    reference = doclink(
                        'builtin-metrics',
                        template='For a list of built-in scorers please refer to {link}'
                    )
                    raise ValueError(
                        'Only built-in scorers are allowed when serializing check instances. '
                        f'{reference}. Scorer name: {k}'
                    )
        return super().config(include_version=include_version)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Return the values of the metrics for the dataset provided in a {metric: value} format."""
        result = {row['Metric'] + '_' + str(row['Class']): row['Value'] for _, row in check_result.value.iterrows()}
        for key in [key for key in result.keys() if key.endswith('_<NA>')]:
            result[key.replace('_<NA>', '')] = result.pop(key)
        return result

    def add_condition_greater_than(self, threshold: float, metrics: Optional[List[str]] = None, class_mode: str = 'all'
                                   ):
        """Add condition - the selected metrics scores are greater than the threshold.

        Parameters
        ----------
        threshold: float
            The threshold that the metrics result should be grater than.
        metrics: List[str]
            The names of the metrics from the check to apply the condition to. If None, runs on all the metrics that
            were calculated in the check.
        class_mode: str, default: 'all'
            The decision rule over the classes, one of 'any', 'all', class name. If 'any', passes if at least one class
            result is above the threshold, if 'all' passes if all the class results are above the threshold,
            class name, passes if the result for this specified class is above the threshold.
        """

        def condition(check_result, metrics_to_check=metrics):
            metrics_to_check = check_result['Metric'].unique() if metrics_to_check is None else metrics_to_check
            metrics_pass = []

            for metric in metrics_to_check:
                if metric not in check_result.Metric.unique():
                    raise DeepchecksValueError(f'The requested metric was not calculated, the metrics calculated in '
                                               f'this check are: {check_result.Metric.unique()}.')

                class_val = check_result[check_result.Metric == metric].groupby('Class').Value
                class_gt = class_val.apply(lambda x: x > threshold)
                if class_mode == 'all':
                    metrics_pass.append(all(class_gt))
                elif class_mode == 'any':
                    metrics_pass.append(any(class_gt))
                elif class_mode in class_val.groups:
                    metrics_pass.append(class_gt[class_val.indices[class_mode]].item())
                else:
                    raise DeepchecksValueError(f'class_mode expected be one of the classes in the check results or any '
                                               f'or all, received {class_mode}.')

            if all(metrics_pass):
                return ConditionResult(ConditionCategory.PASS, 'Passed for all of the metrics.')
            else:
                failed_metrics = ([a for a, b in zip(metrics_to_check, metrics_pass) if not b])
                return ConditionResult(ConditionCategory.FAIL, f'Failed for metrics: {failed_metrics}')

        return self.add_condition(f'Selected metrics scores are greater than {format_number(threshold)}', condition)
