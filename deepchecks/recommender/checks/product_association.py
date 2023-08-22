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
from typing import TypeVar, Union
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from deepchecks.core import CheckResult
from deepchecks.recommender import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck
import numpy as np

__all__ = ['ProductAssociation']

SDP = TypeVar('SDP', bound='ProductAssociation')


class ProductAssociation(SingleDatasetCheck):
    """
    Check for performing non-directional product association analysis based on lift metric.

    It analyzes product associations within a given dataset in a non-directional way.
    It identifies co-occurrences of products within a specified time window (if exists).
    The analysis uses the lift metric to compare probabilities of product co-occurrences.
    Let say the probability of buying product X (e.g., ketchup) in a supermarket is 10%,
    and product Y (e.g., ground beef) is 20%, then if we assume independence, the probability
    of both happening together would be P(X) * P(Y) = 2%.
    However, if the probability of both products occurring together is 8%,resulting in a lift of 4,
    it means that those products are four times more likely to be purchased together
    than if they had no relationship to each other.

    Parameters:
        n_samples (int or None, optional): maximum number of samples to consider from the dataset.
            If set to None, the entire dataset will be used. Default is 1,000,000.
        max_timestamp_delta (int, optional): The maximum time difference (in seconds) allowed
        between two products to be considered as a co-occurrence. Default is 3600 seconds.
        random_state (int, optional): Random seed for reproducibility. Default is 42.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 n_samples: Union[int, None] = 1_000_000,
                 max_timestamp_delta : Union[int, None] = None,
                 random_state: int = 42,
                 **kwargs):

        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.max_timestamp_delta = max_timestamp_delta
        self.random_state = random_state

    def vizualize_graph(self, association_df: pd.DataFrame) -> go.Figure:
        """
        Visualize the association graph.

        Args:
            display_df (DataFrame): The DataFrame prepared for display.

        Returns:
            Figure: The visualization graph.
        """
        net = nx.Graph()

        # Add edges to the graph based on the lift metric above the threshold
        for _, row in association_df.iterrows():
            net.add_edge(row['reference'], row['highly associated items'], weight=row['lift_metric (%)'])
        # Set node positions using a spring layout
        pos = nx.spring_layout(net, k=1.2, seed=32)
        # Create Plotly nodes
        nodes_trace = go.Scatter(
            x=[pos[node][0] for node in net.nodes()],
            y=[pos[node][1] for node in net.nodes()],
            mode='markers+text',
            hovertext=list(net.nodes()),
            textposition='top center',
            hoverinfo='text',
            marker={
                'showscale': True,
                'color': list(dict(net.degree).values()),
                'size': 20,
                'line_width': 2,
                'colorbar_title': 'Node Degree'  # Set the colorbar title

            },


        )

        # Create Plotly edges
        edge_x = []
        edge_y = []
        for edge in net.edges():
            x_0, y_0 = pos[edge[0]]
            x_1, y_1 = pos[edge[1]]
            edge_x.extend([x_0, x_1, None])
            edge_y.extend([y_0, y_1, None])

        edges_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line={'width': 0.5,
                  'color': 'gray'},
            textposition='top center',
            hoverinfo='text',
            mode='lines',
            showlegend=False

        )

        # Combine nodes, edges, and edge labels traces
        fig = go.Figure(data=[edges_trace, nodes_trace])

        # Set plot layout
        fig.update_layout(
            title_text='Item Associations based on Lift Metric',
            showlegend=False,
            hovermode='closest',
            xaxis_title='Layout X-coordinate',  # Set x-axis title
            yaxis_title='Layout Y-coordinate',  # Set y-axis title
        )
        return fig

    def sort_and_combine_items(self, association_df : pd.DataFrame) -> pd.DataFrame:
        """
        Sorts and combines items in the given DataFrame.

        Parameters:
            association_df (pd.DataFrame): DataFrame containing item associations.

        Returns:
            pd.DataFrame: DataFrame with items sorted and combined.
        """
        data = association_df.to_numpy()
        data = np.sort(data, axis=1)
        data = [tuple(row) for row in data]
        return data

    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        """Run check."""
        dataset = context.get_data_by_kind(dataset_kind)

        item_dataset = context.get_item_dataset
        item_column_name = item_dataset.item_column_name
        item_df = item_dataset.data

        user_col = dataset.user_index_name
        item_col = dataset.item_index_name

        interactions = pd.DataFrame(dataset.data)
        # Exclude cold start users by dropping NaN due to the .shift method
        interactions[f'prev_{item_col}'] = interactions.groupby(user_col)[item_col].shift(1)
        interactions[f'prev_{item_col}'] = interactions[f'prev_{item_col}'].astype('Int64').dropna()

        if self.max_timestamp_delta is not None:
            ts_name = dataset.datetime_name
            grouped_interactions = interactions.groupby(user_col)
            interactions[f'prev_{ts_name}'] = grouped_interactions[ts_name].shift(1)
            interactions['delta_sec'] = (interactions[ts_name] - interactions[f'prev_{ts_name}']).dt.total_seconds()
            interactions = interactions[interactions['delta_sec'] < self.max_timestamp_delta]
            interactions = interactions.drop([f'prev_{ts_name}', f'{ts_name}', 'delta_sec'], axis=1)

        association_df = interactions[[f'prev_{item_col}', item_col]].copy()
        association_df['item_combo'] = self.sort_and_combine_items(association_df)

        # Create a new DataFrame to count the non-directional co-occurrence of each pair
        association_counts = association_df.groupby(['item_combo']).size().reset_index(name='co-occurrence')
        # Use the apply method to unpack the tuples into separate columns
        association_counts[f'prev_{item_col}'], association_counts[item_col] = zip(*association_counts['item_combo'])
        # Drop the original 'tuple_column'
        association_counts.drop(columns=['item_combo'], inplace=True)

        prob_df = interactions.groupby(item_col).size().reset_index(name='popularity')
        prob_df['probability'] = prob_df['popularity']/len(interactions)
        prob_dict = prob_df.set_index(item_col)['probability'].to_dict()

        # Function to calculate probability of an item
        def get_probability(item):
            return prob_dict.get(item, 0)

        # Add popularity columns to the DataFrame
        association_counts[f'prev_{item_col}_prb'] = association_counts[f'prev_{item_col}'].apply(get_probability)
        association_counts[f'{item_col}_prb'] = association_counts[item_col].apply(get_probability)
        co_occurence = association_counts['co-occurrence']/len(interactions)
        joint_prob = association_counts[f'prev_{item_col}_prb']*association_counts[f'{item_col}_prb']
        association_counts['lift_metric (%)'] = (co_occurence/joint_prob).astype(np.float16)

        association_counts = association_counts[[f'prev_{item_col}', item_col, 'co-occurrence', 'lift_metric (%)']]
        item_translation = item_df[[item_col, item_column_name]].set_index(item_col)[item_column_name].to_dict()

        association_counts[f'prev_{item_col}'] = association_counts[f'prev_{item_col}'].map(item_translation)
        association_counts[item_col] = association_counts[item_col].map(item_translation)

        association_counts.sort_values(['co-occurrence', 'lift_metric (%)'], ascending=[False, False], inplace=True)
        association_counts.rename(columns={f'prev_{item_col}': 'reference'}, inplace=True)
        association_counts.rename(columns={item_col: 'highly associated items'}, inplace=True)
        display_df = association_counts.head(20)
        fig = self.vizualize_graph(display_df)
        # Show the interactive plot
        text = '''
                This graph illustrates best item association using the lift metric.
                The spring layout visually arranges items in 2D space for representation, while node color highlight
                their significance, indicating the node degree.
                The values, resulting from the layout algorithm in the axis, lack direct physical meaning.
                The absolute values may not carry significant interpretation; instead, the relative distances
                 between nodes matter more for graph visualization.
               '''
        return CheckResult(association_counts, header='Product Association', display=[text, fig])
