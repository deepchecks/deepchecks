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
"""Module containing the single dataset performance check."""
from numbers import Number
from typing import Callable, Dict, List, TypeVar, Union, cast

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.check_utils.single_dataset_performance_base import BaseSingleDatasetPerformance
from deepchecks.tabular import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck

__all__ = ['SingleDatasetPerformance']


SDP = TypeVar('SDP', bound='SingleDatasetPerformance')


class SingleDatasetPerformance(SingleDatasetCheck, BaseSingleDatasetPerformance):
    """Summarize given model performance on a dataset based on selected scorers.

    Parameters
    ----------
    scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
        List of scorers to use. If None, use default scorers.
        Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
    """

    def __init__(self,
                 scorers: Union[List[str], Dict[str, Union[str, Callable]]] = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=False)

        results = []
        classes = dataset.classes
        label = cast(pd.Series, dataset.label_col)
        n_samples = label.groupby(label).count()
        for scorer in scorers:
            scorer_value = scorer(model, dataset)
            if isinstance(scorer_value, Number):
                results.append([pd.NA, scorer.name, scorer_value, len(label)])
            else:
                results.extend(
                    [[class_name, scorer.name, class_score, n_samples[class_name]]
                     for class_score, class_name in zip(scorer_value, classes)])
        results_df = pd.DataFrame(results, columns=['Class', 'Metric', 'Value', 'Number of samples'])

        if context.with_display:
            display = [results_df]
        else:
            display = []

        return CheckResult(results_df, header='Single Dataset Performance', display=display)