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
"""Module containing common WholeDatasetDriftCheck (domain classifier drift) utils."""
import warnings
from typing import Container, List

import pandas as pd
import plotly.graph_objects as go

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa # pylint: disable=unused-import

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from deepchecks.tabular import Dataset
from deepchecks.utils.dataframes import floatify_dataframe
from deepchecks.utils.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.utils.distribution.rare_category_encoder import RareCategoryEncoder
from deepchecks.utils.features import N_TOP_MESSAGE, calculate_feature_importance_or_none
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable


def run_whole_dataset_drift(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame,
                            numerical_features: List[Hashable], cat_features: List[Hashable], sample_size: int,
                            random_state: int, test_size: float, n_top_columns: int, min_feature_importance: float,
                            max_num_categories_for_display: int, show_categories_by: str,
                            min_meaningful_drift_score: float,
                            with_display: bool):
    """Calculate whole dataset drift."""
    train_sample_df = train_dataframe.sample(sample_size, random_state=random_state)[numerical_features + cat_features]
    test_sample_df = test_dataframe.sample(sample_size, random_state=random_state)[numerical_features + cat_features]

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_df = pd.concat([train_sample_df, test_sample_df])
    domain_class_df[cat_features] = RareCategoryEncoder(254).fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_df[cat_features] = OrdinalEncoder().fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels,
                                                        random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = HistGradientBoostingClassifier(max_depth=2, max_iter=10, random_state=random_state,
                                                       categorical_features=[x in cat_features for x in
                                                                             domain_class_df.columns])
    domain_classifier.fit(x_train, y_train)

    y_test.name = 'belongs_to_test'
    domain_test_dataset = Dataset(pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
                                  cat_features=cat_features, label='belongs_to_test')

    # calculate feature importance of domain_classifier, containing the information which features separate
    # the dataset best.
    fi, importance_type = calculate_feature_importance_or_none(
        domain_classifier,
        domain_test_dataset,
        force_permutation=True,
        permutation_kwargs={'n_repeats': 10,
                            'random_state': random_state,
                            'timeout': 120,
                            'skip_messages': True}
    )

    fi = fi.sort_values(ascending=False) if fi is not None else None

    domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {
        'domain_classifier_auc': domain_classifier_auc,
        'domain_classifier_drift_score': drift_score,
        'domain_classifier_feature_importance': fi.to_dict() if fi is not None else {},
    }

    feature_importance_note = f"""
    <span>
    The percents of explained dataset difference are the importance values for the feature calculated
    using `{importance_type}`.
    </span><br><br>
    """

    if with_display and fi is not None and drift_score > min_meaningful_drift_score:
        top_fi = fi.head(n_top_columns)
        top_fi = top_fi.loc[top_fi > min_feature_importance]
    else:
        top_fi = None

    if top_fi is not None and len(top_fi):
        score = values_dict['domain_classifier_drift_score']

        displays = [
            feature_importance_note,
            build_drift_plot(score),
            '<h3>Main features contributing to drift</h3>',
            N_TOP_MESSAGE % n_top_columns,
            *(
                display_dist(
                    train_sample_df[feature],
                    test_sample_df[feature],
                    top_fi,
                    cat_features,
                    max_num_categories_for_display,
                    show_categories_by)
                for feature in top_fi.index
            )
        ]
    else:
        displays = None

    return values_dict, displays


def auc_to_drift_score(auc: float) -> float:
    """Calculate the drift score, which is 2*auc - 1, with auc being the auc of the Domain Classifier.

    Parameters
    ----------
    auc : float
        auc of the Domain Classifier
    """
    return max(2 * auc - 1, 0)


def build_drift_plot(score):
    """Build traffic light drift plot."""
    bar_traces, x_axis, y_axis = drift_score_bar_traces(score)
    x_axis['title'] = 'Drift score'
    drift_plot = go.Figure(layout=dict(
        title='Drift Score - Whole Dataset Total',
        xaxis=x_axis,
        yaxis=y_axis,
        height=200
    ))

    drift_plot.add_traces(bar_traces)
    return drift_plot


def display_dist(
        train_column: pd.Series,
        test_column: pd.Series,
        fi: pd.Series,
        cat_features: Container[str],
        max_num_categories: int,
        show_categories_by: str
):
    """Create a distribution comparison plot for the given columns."""
    column_name = train_column.name or ''
    column_fi = fi.loc[column_name]
    title = f'Feature: {column_name} - Explains {format_percent(column_fi)} of dataset difference'

    dist_traces, xaxis_layout, yaxis_layout = feature_distribution_traces(
        train_column.dropna(),
        test_column.dropna(),
        column_name,
        is_categorical=column_name in cat_features,
        max_num_categories=max_num_categories,
        show_categories_by=show_categories_by
    )

    all_categories = list(set(train_column).union(set(test_column)))
    add_footnote = column_name in cat_features and len(all_categories) > max_num_categories

    if not add_footnote:
        fig = go.Figure()
        fig.add_traces(dist_traces)
    else:

        if show_categories_by == 'train_largest':
            categories_order_description = 'largest categories (by train)'
        elif show_categories_by == 'test_largest':
            categories_order_description = 'largest categories (by test)'
        elif show_categories_by == 'largest_difference':
            categories_order_description = 'largest difference between categories'
        else:
            raise ValueError(f'Unsupported "show_categories_by" value - {show_categories_by}')

        train_data_percents = dist_traces[0].y.sum()
        test_data_percents = dist_traces[1].y.sum()

        annotation = (
            f'* Showing the top {max_num_categories} {categories_order_description} out of '
            f'total {len(all_categories)} categories.'
            f'<br>Shown data is {format_percent(train_data_percents)} of train data and '
            f'{format_percent(test_data_percents)} of test data.'
        )

        fig = (
            go.Figure()
            .add_traces(dist_traces)
            .add_annotation(
                x=0, y=-0.4,
                showarrow=False,
                xref='paper',
                yref='paper',
                xanchor='left',
                text=annotation)
        )

    return fig.update_layout(go.Layout(
        title=title,
        xaxis=xaxis_layout,
        yaxis=yaxis_layout,
        legend=dict(
            title='Dataset',
            yanchor='top',
            y=0.9,
            xanchor='left'),
        height=300
    ))
