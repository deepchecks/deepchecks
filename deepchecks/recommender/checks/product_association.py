from typing import TYPE_CHECKING, Callable, Dict, List, Mapping, Optional, TypeVar, Union, cast
import plotly.graph_objects as go
import networkx as nx
import pandas as pd
from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.recommender import Context
from deepchecks.tabular.base_checks import SingleDatasetCheck

from deepchecks.utils.docref import doclink
from deepchecks.utils.strings import format_number

if TYPE_CHECKING:
    from deepchecks.core.checks import CheckConfig

__all__ = ['ProductAssociation']


SDP = TypeVar('SDP', bound='ProductAssociation')

class ProductAssociation(SingleDatasetCheck):
    """    
    Check for performing non-directional product association analysis based on lift metric. It analyzes product associations within a given dataset in a non-directional way. It identifies co-occurrences of products within a specified time window (if exists) to reveal potential product associations, used for recommender systems.

    The analysis uses the lift metric to compare probabilities of product co-occurrences. Let say the
    probability of buying product X (e.g., ketchup) in a supermarket is 10%, and product Y (e.g., ground beef)
    is 20%, then if we assume independence, the probability of both happening together would be P(X) * P(Y) = 2%.
    However, if the probability of both products occurring together is 8%, resulting in a lift of 8 / 2 = 4,
    it means that those products are four times more likely to be purchased together than if they had
    no relationship to each other.

    Parameters:
        n_samples (int or None, optional): The maximum number of samples to consider from the dataset.
            If set to None, the entire dataset will be used. Default is 1,000,000.
        max_timestamp_delta (int, optional): The maximum time difference (in seconds) allowed between
            two product occurrences to be considered as a co-occurrence. Default is 3600 seconds (1 hour).
        random_state (int, optional): Random seed for reproducibility. Default is 42.
        **kwargs: Additional keyword arguments.
    """

    def __init__(self,
                 n_samples: Union[int,None] = 1_000_000,
                 max_timestamp_delta : Union[int, None] = None,
                 random_state: int = 42,
                 **kwargs
    ):
        super().__init__(**kwargs)
        self.n_samples = n_samples
        self.max_timestamp_delta = max_timestamp_delta
        self.random_state = random_state


    def run_logic(self, context: Context, dataset_kind) -> CheckResult:
        
        dataset = context.get_data_by_kind(dataset_kind)#.sample(self.n_samples, random_state=self.random_state)
        
        item_dataset = context._item_dataset
        item_column_name = item_dataset.item_column_name
        item_df = item_dataset.data
        
        user_col = dataset.user_index_name
        item_col = dataset.item_index_name

        filtered_interactions = pd.DataFrame(dataset.data)
        # Ommit cold start users by droping NaN because of the .shift
        filtered_interactions[f'prev_{item_col}'] = filtered_interactions.groupby(user_col)[item_col].shift(1).astype("Int64").dropna()

        if self.max_timestamp_delta is not None:
            datetime_name = dataset.datetime_name
            filtered_interactions[f'prev_{datetime_name}'] = filtered_interactions.groupby(user_col)[datetime_name].shift(1)
            filtered_interactions[f'delta_{datetime_name}_seconds'] = (filtered_interactions[datetime_name] - filtered_interactions[f'prev_{datetime_name}']).dt.total_seconds()
            filtered_interactions = filtered_interactions[filtered_interactions['delta_timestamp_seconds'] < self.max_timestamp_delta].drop([f'prev_{datetime_name}',f'{datetime_name}'],axis=1)
                 
        products_association_df = filtered_interactions[[f'prev_{item_col}', item_col]].copy()                                                                                                                               
        products_association_df['item_combination'] = products_association_df.apply(lambda x: tuple(sorted(x)), axis=1)

        # Create a new DataFrame to count the non-directional co-occurrence of each pair
        association_counts = products_association_df.groupby(['item_combination']).size().reset_index(name='co-occurence')
        # Use the apply method to unpack the tuples into separate columns
        association_counts[f'prev_{item_col}'], association_counts[item_col] = zip(*association_counts['item_combination'])
        # Drop the original 'tuple_column' 
        association_counts.drop(columns=['item_combination'], inplace=True)

        prob_df = filtered_interactions.groupby(item_col).size().reset_index(name='popularity')
        prob_df['probability'] = prob_df['popularity']/len(filtered_interactions)
        prob_dict = prob_df.set_index(item_col)['probability'].to_dict()
        
        # Function to calculate probability of an item
        def get_probability(item):
            return prob_dict.get(item, 0)

        # Add popularity columns to the DataFrame
        association_counts[f'prev_{item_col}_prb'] = association_counts[f'prev_{item_col}'].apply(get_probability)
        association_counts[f'{item_col}_prb'] = association_counts[item_col].apply(get_probability)
        association_counts['lift_metric (%)'] = association_counts['co-occurence']/(association_counts[f'prev_{item_col}_prb']*association_counts[f'{item_col}_prb']*len(filtered_interactions)).astype(np.float16)
        
        association_counts = association_counts[[f'prev_{item_col}', item_col,'co-occurence','lift_metric (%)']]
        movie_translation = item_df[[item_col,item_column_name]].set_index(item_col)[item_column_name].to_dict()
        
        association_counts[f'prev_{item_col}']= association_counts[f'prev_{item_col}'].map(movie_translation)
        association_counts[item_col]= association_counts[item_col].map(movie_translation)

        association_counts = association_counts.sort_values(['co-occurence','lift_metric (%)'],ascending=[False,False]).rename({f'prev_{item_col}':'reference',
        item_col:'possible recommandation'},axis=1)
        display_df = association_counts.head(20)
        # Create a network graph
        G = nx.Graph()

        # Add edges to the graph based on the lift metric above the threshold
        for _, row in display_df.iterrows():
                G.add_edge(row['reference'], row['possible recommandation'], weight=row['lift_metric (%)'])

        # Set node positions using a spring layout
        pos = nx.spring_layout(G,k=1.2,seed=32)


        # Create Plotly nodes
        nodes_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hovertext=list(G.nodes()),
            textposition='top center',
            hoverinfo='text',
            marker=dict(
                showscale=False,
                color=list(dict(G.degree).values()),
                size=20,
                line_width=2
            )
        )

        # Create Plotly edges
        edge_x = []
        edge_y = []
        edge_weights = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
            edge_weights.append(G.edges[edge]['weight'])

        edges_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color='gray'),
            textposition='top center',
            hoverinfo='text',  # Show hovertext
            mode='lines'
        )

        # Combine nodes, edges, and edge labels traces
        fig = go.Figure(data=[edges_trace, nodes_trace])

        # Set plot layout
        fig.update_layout(
            title_text='Item Associations based on Lift Metric',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
        )

        # Show the interactive plot

        return CheckResult(association_counts, header='Product Association', display=[fig])
