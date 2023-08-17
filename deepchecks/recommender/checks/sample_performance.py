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
"""Module containing the single dataset performance check for recommender system."""
from typing import Callable, List, Mapping, Optional, TypeVar, Union
import plotly.express as px

import pandas as pd

from deepchecks.core import CheckResult
from deepchecks.recommender import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck

__all__ = ['SamplePerformance']

SDP = TypeVar('SDP', bound='SamplePerformance')


class SamplePerformance(SingleDatasetCheck):
    """Summarize given recommender system model performance based on selected scorers.

    Parameters
    ----------
    scorers: Union[Mapping[str, Union[str, Callable]], List[str]], default: None
        Scorers to override the default scorers, find more about the supported formats at
        https://docs.deepchecks.com/stable/user-guide/general/metrics_guide.html
    n_samples : int , default: 1_000_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(self,
                 scorers: Optional[Union[Mapping[str, Union[str, Callable]], List[str]]] = None,
                 n_samples: Union[int, None] = 1_000_000,
                 random_state: int = 42,
                 **kwargs):
        super().__init__(**kwargs)
        self.scorers = scorers
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples,
                                                                random_state=self.random_state)
        model = context.model
        scorers = context.get_scorers(self.scorers, use_avg_defaults=True)

        results = []
        for scorer in scorers:
            scorer_value = scorer(model, dataset)
            results.append([scorer.name, scorer_value])
        results_df = pd.DataFrame(results, columns=['Metric', 'Value'])

        if context.with_display:
            fig = px.bar(results_df, y='Value', x='Metric')

        return CheckResult(results_df, header='Sample Performance', display=fig)
