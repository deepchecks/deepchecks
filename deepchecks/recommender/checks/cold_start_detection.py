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
"""Module of ColdStartDetection check."""
import pandas as pd
from deepchecks.core import CheckResult
from deepchecks.tabular import SingleDatasetCheck
from deepchecks.recommender import Context, InteractionDataset
from deepchecks.utils.dataframes import select_from_dataframe
import plotly.express as px

__all__ = ['ColdStartDetection']


class ColdStartDetection(SingleDatasetCheck):
    """Retrieve cold start users and items, which are new entities with no historical data.

    The proportion of cold start entities increases uncertainty and difficulty in personalization.
    It may result in reduced user engagement, missed opportunities, and limited user understanding.
    The recommender model should be trained on the remaining historical data.

    Parameters
    ----------
    n_samples : int , default: 5_000
        number of samples to use for this check.
    random_state : int, default: 42
        random seed for all check internals.
    """

    def __init__(
            self,
            n_samples: int = 1_000_000,
            random_state: int = 42,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.random_state = random_state

    def run_logic(self, context: Context, dataset_kind : InteractionDataset) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind).sample(self.n_samples, self.random_state)
        interaction_df = select_from_dataframe(dataset.data)
        # Group the DataFrame by user and item
        grouped_users = interaction_df.groupby(dataset.user_index_name).size()
        grouped_items = interaction_df.groupby(dataset.item_index_name).size()
        cold_start_users = grouped_users[grouped_users == 1]
        cold_start_items = grouped_items[grouped_items == 1]

        # Calculate cold start proportions
        cold_start_users_pct = len(cold_start_users) / len(grouped_users)
        cold_start_items_pct = len(cold_start_items) / len(grouped_items)

        # Create results DataFrame
        results_df = pd.DataFrame([
            ['Users', 100 * cold_start_users_pct],
            ['Items', 100 * cold_start_items_pct]
        ], columns=['Entity', 'Cold Start Proportion (%)'])
        if context.with_display:
            fig = px.bar(results_df, y='Cold Start Proportion (%)', x='Entity')
            fig.update_layout(title='Cold Start Proportion per entity (%)')
            fig.update_layout(bargap=0.2)
            fig.update_traces(width=0.6)
            fig.update_layout(yaxis_range=[0, 100])  # Set the range of y-axis to 0 and 100

        return CheckResult(cold_start_users_pct, display=fig)
