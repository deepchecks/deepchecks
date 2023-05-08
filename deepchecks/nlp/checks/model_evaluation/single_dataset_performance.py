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
"""Module containing the single dataset performance check."""
from numbers import Number
from typing import Callable, Dict, List, Union

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.core.check_utils.single_dataset_performance_base import BaseSingleDatasetPerformance
from deepchecks.nlp.base_checks import SingleDatasetCheck
from deepchecks.nlp.context import Context
from deepchecks.nlp.metric_utils.scorers import infer_on_text_data

__all__ = ['SingleDatasetPerformance']


class SingleDatasetPerformance(SingleDatasetCheck, BaseSingleDatasetPerformance):
    """Summarize given model performance on a dataset based on selected scorers.

    Parameters
    ----------
    scorers : Union[List[str], Dict[str, Union[str, Callable]]], default: None
        List of scorers to use. If None, use default scorers.
        Scorers can be supplied as a list of scorer names or as a dictionary of names and functions.
    max_rows_to_display : int, default: 15
        Maximum number of rows to display in the check result.
    n_samples : int , default: 10_000
        Maximum number of samples to use for this check.
    """

    def __init__(self,
                 scorers: Union[List[str], Dict[str, Union[str, Callable]]] = None,
                 max_rows_to_display: int = 15,
                 n_samples: int = 10_000,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.max_rows_to_display = max_rows_to_display
        self.n_samples = n_samples

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)
        dataset = dataset.sample(self.n_samples, random_state=context.random_state)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=False)

        results = []
        for scorer in scorers:
            scorer_value = infer_on_text_data(scorer, model, dataset)
            if isinstance(scorer_value, Number):
                results.append([pd.NA, scorer.name, scorer_value])
            else:
                results.extend(
                    [[class_name, scorer.name, class_score]
                     for class_name, class_score in scorer_value.items()])
        results_df = pd.DataFrame(results, columns=['Class', 'Metric', 'Value'])

        if context.with_display:
            if len(results_df) > self.max_rows_to_display:
                display = [results_df.iloc[:self.max_rows_to_display, :],
                           '<p style="font-size:0.9em;line-height:1;"><i>'
                           f'* Only showing first {self.max_rows_to_display} rows.']
            else:
                display = [results_df]
        else:
            display = []

        return CheckResult(results_df, header='Single Dataset Performance', display=display)
