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
"""Module containing common MultivariateDrift Check (domain classifier drift) utils."""
import numpy as np
import warnings
from typing import Container, List, Tuple, Dict

import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa # pylint: disable=unused-import

from sklearn.ensemble import HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

from deepchecks.tabular import Dataset
from deepchecks.tabular.utils.feature_importance import N_TOP_MESSAGE, calculate_feature_importance_or_none
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.dataframes import floatify_dataframe
from deepchecks.utils.distribution.drift import get_drift_plot_sidenote
from deepchecks.utils.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.utils.distribution.rare_category_encoder import RareCategoryEncoder
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable


def run_multivariable_drift(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame,
                            numerical_features: List[Hashable], cat_features: List[Hashable], sample_size: int,
                            random_state: int, test_size: float, n_top_columns: int, min_feature_importance: float,
                            max_num_categories_for_display: int, show_categories_by: str,
                            min_meaningful_drift_score: float,
                            with_display: bool,
                            dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES,
                            feature_importance_timeout: int = 120,
                            ):
    """Calculate multivariable drift."""
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
        model_classes=[0, 1],
        observed_classes=[0, 1],
        task_type=TaskType.BINARY,
        force_permutation=True,
        permutation_kwargs={'n_repeats': 10,
                            'random_state': random_state,
                            'timeout': feature_importance_timeout,
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
            get_drift_plot_sidenote(max_num_categories_for_display, show_categories_by),
            *(
                display_dist(
                    train_sample_df[feature],
                    test_sample_df[feature],
                    top_fi,
                    cat_features,
                    max_num_categories_for_display,
                    show_categories_by,
                    dataset_names)
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
        title='Drift Score - Multivariable',
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
        show_categories_by: str,
        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES
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
        show_categories_by=show_categories_by,
        dataset_names=dataset_names
    )

    fig = go.Figure()
    fig.add_traces(dist_traces)

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


def run_multivariable_drift_for_embeddings(train_embeddings: pd.DataFrame, test_embeddings: pd.DataFrame, train_dataset,
                                           test_dataset,  # Type TextData but I don't want to import
                                           numerical_features: List[Hashable], cat_features: List[Hashable],
                                           sample_size: int, random_state: int, test_size: float, n_top_columns: int,
                                           min_feature_importance: float, min_meaningful_drift_score: float,
                                           num_samples_in_display: int, with_display: bool,
                                           dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES,
                                           train_indexes_to_highlight: List[int] = [],
                                           test_indexes_to_highlight: List[int] = [], ):
    """Calculate multivariable drift."""
    # TODO: Prototype, go over and make sure code+docs+tests are good

    import random
    random.seed(random_state)

    train_sample_df = train_embeddings.sample(sample_size, random_state=random_state)[numerical_features + cat_features]
    test_sample_df = test_embeddings.sample(sample_size, random_state=random_state)[numerical_features + cat_features]

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_df = pd.concat([train_sample_df, test_sample_df])
    domain_class_df[cat_features] = RareCategoryEncoder(254).fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_df[cat_features] = OrdinalEncoder().fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

    from sklearn.decomposition import PCA
    from umap import UMAP

    # reducer = PCA(n_components=30, random_state=42)
    reducer = UMAP(init='random', random_state=42, n_components=10, n_neighbors=15, min_dist=0.1)
    domain_class_df = pd.DataFrame(reducer.fit_transform(domain_class_df), index=domain_class_df.index)

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels, random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=2, random_state=random_state)
    domain_classifier.fit(x_train, y_train)

    y_test.name = 'belongs_to_test'

    # calculate feature importance of domain_classifier, containing the information which features separate
    # the dataset best.
    fi = pd.Series(domain_classifier.feature_importances_, index=x_train.columns)
    importance_type = 'internal_feature_importance'  # TODO is this the correct term?

    fi = fi.sort_values(ascending=False) if fi is not None else None

    domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {'domain_classifier_auc': domain_classifier_auc, 'domain_classifier_drift_score': drift_score,
                   'domain_classifier_feature_importance': fi.to_dict() if fi is not None else {}, }

    feature_importance_note = f"""
    <span>
    The percents of explained dataset difference are the importance values for the feature calculated
    using `{importance_type}`.
    </span><br><br>
    """

    if with_display and fi is not None and drift_score > min_meaningful_drift_score:
        top_fi = fi.head(n_top_columns)
        top_fi_under_threshold = top_fi[top_fi < min_feature_importance]
        if not top_fi_under_threshold.shape[0] == 0:
            top_fi = top_fi_under_threshold
    else:
        top_fi = None

    if drift_score > min_meaningful_drift_score:
        train_embeddings = pd.DataFrame(reducer.transform(train_embeddings), index=train_embeddings.index)
        test_embeddings = pd.DataFrame(reducer.transform(test_embeddings), index=test_embeddings.index)

        # Sample data before display calculations
        num_samples_in_display = min(num_samples_in_display, sample_size)
        samples_for_display = np.random.choice(range(len(train_dataset)), size=num_samples_in_display, replace=False).tolist()
        train_dataset_for_display = train_dataset.copy(samples_for_display)
        train_embeddings = train_embeddings.iloc[samples_for_display]
        # train_indexes_to_highlight = [x for x in train_indexes_to_highlight if x in train_dataset_for_display.index]
        samples_for_display = np.random.choice(range(len(test_dataset)), size=num_samples_in_display, replace=False).tolist()
        test_dataset_for_display = test_dataset.copy(samples_for_display)
        test_embeddings = test_embeddings.iloc[samples_for_display]
        # test_indexes_to_highlight = [x for x in test_indexes_to_highlight if x in test_dataset_for_display.index]

        # Calculate display
        embeddings_for_display = pd.concat([train_embeddings, test_embeddings])
        # embeddings_for_display = pd.DataFrame(pca.transform(embeddings_for_display), index=embeddings_for_display.index)
        domain_classifier_probas = domain_classifier.predict_proba(floatify_dataframe(embeddings_for_display))[:, 1]

        displays = [feature_importance_note, build_drift_plot(drift_score),
                    # display_embeddings_only(train_embeddings=train_embeddings, test_embeddings=test_embeddings,
                    #                         top_fi_embeddings=top_fi, train_dataset=train_dataset_for_display,
                    #                         test_dataset=test_dataset_for_display,
                    #                         train_indexes_to_highlight=train_indexes_to_highlight,
                    #                         test_indexes_to_highlight=test_indexes_to_highlight),
                    # display_embeddings_with_clusters_proba_as_target(train_embeddings=train_embeddings,
                    #                                                  test_embeddings=test_embeddings,
                    #                                                  train_dataset=train_dataset_for_display,
                    #                                                  test_dataset=test_dataset_for_display, domain_classifier_fi=fi,
                    #                                                  train_indexes_to_highlight=train_indexes_to_highlight,
                    #                                                  test_indexes_to_highlight=test_indexes_to_highlight,
                    #                                                  domain_classifier_probas=domain_classifier_probas),
                    # display_embeddings_with_clusters_by_nodes_with_onehot(train_embeddings=train_embeddings,
                    #                                                       test_embeddings=test_embeddings,
                    #                                                       train_dataset=train_dataset_for_display,
                    #                                                       test_dataset=test_dataset_for_display,
                    #                                                       domain_classifier_fi=fi,
                    #                                                       train_indexes_to_highlight=train_indexes_to_highlight,
                    #                                                       test_indexes_to_highlight=test_indexes_to_highlight),
                    # display_embeddings_proba_as_axis(domain_classifier_probas=domain_classifier_probas,
                    #                                  train_embeddings=train_embeddings, test_embeddings=test_embeddings,
                    #                                  top_fi_embeddings=top_fi, train_dataset=train_dataset_for_display,
                    #                                  test_dataset=test_dataset_for_display,
                    #                                  train_indexes_to_highlight=train_indexes_to_highlight,
                    #                                  test_indexes_to_highlight=test_indexes_to_highlight),
                    display_embeddings_proba_as_axis_with_nodes(
                        domain_classifier_probas=domain_classifier_probas,
                        train_embeddings=train_embeddings, test_embeddings=test_embeddings,
                        top_fi_embeddings=top_fi, train_dataset=train_dataset_for_display,
                        test_dataset=test_dataset_for_display,
                        train_indexes_to_highlight=train_indexes_to_highlight,
                        test_indexes_to_highlight=test_indexes_to_highlight),
                    ]
    else:
        displays = None

    return values_dict, displays


def _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                         train_indexes_to_highlight, indexes_per_node: Dict[int, List[str]] = None):
    import plotly.express as px
    save_for_later = plot_data.copy()
    plot_data['dataset'] = ['train_full'] * len(train_dataset.get_original_text_indexes()) + ['test_full'] * len(test_dataset.get_original_text_indexes())
    if train_dataset.has_label():
        plot_data['label'] = np.concatenate([train_dataset.label_for_display, test_dataset.label_for_display])
    else:
        plot_data['label'] = None
    plot_data['sample'] = np.concatenate([train_dataset.text, test_dataset.text])
    plot_data['sample'] = plot_data['sample'].apply(clean_sample)

    # Only keep relevant indexes
    # plot_data.index = train_dataset.index + test_dataset.index
    if indexes_per_node is None:
        train_to_add = plot_data[plot_data.index.isin(train_indexes_to_highlight)].copy()
        train_to_add['dataset'] = 'classes_only_in_train'
        test_to_add = plot_data[plot_data.index.isin(test_indexes_to_highlight)].copy()
        test_to_add['dataset'] = 'classes_only_in_test'
        plot_data = pd.concat([plot_data, train_to_add, test_to_add], ignore_index=True)
    else:
        stuff_to_add = []
        for node_id, indexes in indexes_per_node.items():
            print(f'Node {node_id} has {len(indexes)} samples')
            to_add = plot_data[plot_data.index.isin(indexes)].copy()
            print('Percent of train in node is {}'.format(to_add['dataset'].value_counts()['train_full'] / len(to_add)))
            # print(to_add['label'].value_counts().to_dict())
            to_add['dataset'] = f'node_{node_id}'
            stuff_to_add.append(to_add)
        plot_data = pd.concat([plot_data] + stuff_to_add, ignore_index=True)

    fig = px.scatter(plot_data, x=1, y=0, color='dataset', hover_data=['label', 'sample'], hover_name='dataset',
                     title=plot_title, height=1000,  # color_discrete_sequence=['red', 'green', 'blue', 'orange'],
                     width=1000, opacity=0.4)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig


def display_embeddings_only(train_embeddings, test_embeddings, top_fi_embeddings, train_dataset, test_dataset,
                            train_indexes_to_highlight: List[int], test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    if top_fi_embeddings.shape[0] == 1:
        return ''

    from umap import UMAP
    # from sklearn.decomposition import PCA

    method = 'UMAP'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    top_fi_embeddings = top_fi_embeddings.index.values

    if top_fi_embeddings.shape[0] == 2: # Need to fix for case == 1
        reduced_embeddings = embeddings.loc[:, top_fi_embeddings].values
        plot_title = f'Embeddings in 2D (not using {method} as top features == {top_fi_embeddings.shape[0]}'

    else:
        reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
        # reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
        plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} features'

    plot_data = pd.DataFrame(reduced_embeddings)
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight)


def display_embeddings_with_clusters_by_nodes(train_embeddings, test_embeddings, train_dataset, test_dataset,
                                              train_indexes_to_highlight: List[int],
                                              test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    from umap import UMAP
    # from sklearn.decomposition import PCA

    method = 'UMAP'
    # method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=30, min_samples_leaf=50, random_state=42)
    domain_class_labels = pd.Series([0] * len(train_embeddings) + [1] * len(test_embeddings))

    classifier.fit(embeddings, domain_class_labels)

    fi = pd.Series(classifier.feature_importances_, index=train_embeddings.columns)

    fi = fi.sort_values(ascending=False) if fi is not None else None

    top_fi_embeddings = fi.head(20)
    top_fi_embeddings = top_fi_embeddings.loc[top_fi_embeddings > 0.01]

    top_fi_embeddings = top_fi_embeddings.index.values

    domain_classifier_nodes = classifier.apply(embeddings)

    # reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    reduced_embeddings = UMAP(n_components=2, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings],
                                                                             y=domain_classifier_nodes)

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} features with tree-based nodes as a target'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight)


def display_embeddings_with_clusters_proba_as_target(train_embeddings, test_embeddings, train_dataset, test_dataset,
                                                     domain_classifier_fi, train_indexes_to_highlight: List[int],
                                                     test_indexes_to_highlight: List[int], domain_classifier_probas):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    from umap import UMAP

    method = 'UMAP'
    # method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])
    top_fi_embeddings = domain_classifier_fi.head(20)
    top_fi_embeddings = top_fi_embeddings.loc[top_fi_embeddings > 0.01].index.values
    data_to_reduce = embeddings.loc[:, top_fi_embeddings]
    reduced_data = UMAP(n_components=2, random_state=42).fit_transform(data_to_reduce, y=domain_classifier_probas)

    plot_data = pd.DataFrame(reduced_data)
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} ' \
                 f'features with proba as target'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight)


def display_embeddings_with_clusters_by_nodes_with_onehot(train_embeddings, test_embeddings, train_dataset,
                                                          test_dataset, domain_classifier_fi,
                                                          train_indexes_to_highlight: List[int],
                                                          test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    from umap import UMAP
    from sklearn.tree import DecisionTreeClassifier

    method = 'UMAP'
    # method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])
    domain_class_labels = pd.Series([0] * len(train_embeddings) + [1] * len(test_embeddings))
    top_fi_embeddings = domain_classifier_fi.head(10)
    top_fi_embeddings = top_fi_embeddings.loc[top_fi_embeddings > 0.01].index.values

    train, test, train_labels, test_labels = train_test_split(embeddings.loc[:, top_fi_embeddings], domain_class_labels,
                                                              test_size=0.2, random_state=42)
    min_cluster_size = max(50, int(len(train) * 0.04))
    classifier = DecisionTreeClassifier(max_depth=30, min_samples_leaf=min_cluster_size, random_state=42,
                                        criterion='entropy')
    classifier.fit(train, train_labels)
    classifier_auc = roc_auc_score(test_labels, classifier.predict_proba(test)[:, 1])
    print(f'Classifier AUC: {classifier_auc}')

    tree_node_values = test_labels.groupby(classifier.apply(test)).mean()
    interesting_nodes = tree_node_values[tree_node_values < 0.25].index.tolist() + tree_node_values[
        tree_node_values > 0.75].index.tolist()
    print(f'Number of interesting nodes: {len(interesting_nodes)}')

    train_node_ids = pd.Series((x if x in interesting_nodes else -1 for x in classifier.apply(train)),
                               index=train.index)
    test_node_ids = pd.Series((x if x in interesting_nodes else -1 for x in classifier.apply(test)), index=test.index)
    node_per_sample = pd.concat([train_node_ids, test_node_ids])
    one_hot_node_data = pd.get_dummies(node_per_sample, dtype='float').iloc[:, 1:] * ([0.01] * len(interesting_nodes))

    data_to_reduce = embeddings.loc[:, top_fi_embeddings].join(one_hot_node_data)
    reduced_data = UMAP(n_components=2, random_state=42).fit_transform(data_to_reduce)

    plot_data = pd.DataFrame(reduced_data)
    nodes_ids = sorted(node_per_sample.unique())[1:]
    nodes_to_highlight = {x: list(node_per_sample[node_per_sample == x].index) for x in nodes_ids}
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} ' \
                 f'features with one-hot-encoded top {len(interesting_nodes)} nodes as additional embeddings'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight, nodes_to_highlight)


def display_embeddings_proba_as_axis(domain_classifier_probas, train_embeddings, test_embeddings, top_fi_embeddings,
                                     train_dataset, test_dataset, train_indexes_to_highlight: List[int],
                                     test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    # from umap import UMAP
    from sklearn.decomposition import PCA

    # method = 'UMAP'
    method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    top_fi_embeddings = top_fi_embeddings.index.values

    # reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    reduced_embeddings = PCA(n_components=1, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_data[1] = domain_classifier_probas
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} features to 1 dimension with probas as y axis'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight)


def display_embeddings_proba_as_axis_with_nodes(domain_classifier_probas, train_embeddings, test_embeddings,
                                                top_fi_embeddings,
                                                train_dataset, test_dataset, train_indexes_to_highlight: List[int],
                                                test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    # from umap import UMAP
    from sklearn.decomposition import PCA

    # method = 'UMAP'
    method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    top_fi_embeddings = top_fi_embeddings.index.values

    domain_class_labels = pd.Series([0] * len(train_embeddings) + [1] * len(test_embeddings))

    # reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    reduced_embeddings = PCA(n_components=1, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])


    plot_data = pd.DataFrame(reduced_embeddings)
    plot_data[1] = domain_classifier_probas

    # train, test, train_labels, test_labels = train_test_split(plot_data, domain_class_labels,
    #                                                           test_size=0.2, random_state=42)
    # min_cluster_size = max(50, int(len(train) * 0.04))
    # classifier = DecisionTreeClassifier(max_depth=30, min_samples_leaf=min_cluster_size, random_state=42,
    #                                     criterion='entropy')
    # classifier.fit(train, train_labels)
    # classifier_auc = roc_auc_score(test_labels, classifier.predict_proba(test)[:, 1])
    # print(f'Classifier AUC: {classifier_auc}')
    #
    # tree_node_values = test_labels.groupby(classifier.apply(test)).mean()

    min_cluster_size = max(50, int(len(plot_data) * 0.04))
    classifier = DecisionTreeClassifier(max_depth=30, min_samples_leaf=min_cluster_size, random_state=42,
                                        criterion='entropy')
    classifier.fit(plot_data, domain_class_labels) # TODO: Skip graph if no meaningful nodes
    # classifier_auc = roc_auc_score(test_labels, classifier.predict_proba(test)[:, 1])
    # print(f'Classifier AUC: {classifier_auc}')

    tree_node_values = domain_class_labels.groupby(classifier.apply(plot_data)).mean()

    interesting_nodes = tree_node_values[tree_node_values < 0.25].index.tolist() + tree_node_values[
        tree_node_values > 0.75].index.tolist()
    print(f'Number of interesting nodes: {len(interesting_nodes)}')

    node_per_sample = pd.Series((x if x in interesting_nodes else -1 for x in classifier.apply(plot_data)), index=plot_data.index)

    nodes_ids = sorted(node_per_sample.unique())[1:]
    nodes_to_highlight = {x: list(node_per_sample[node_per_sample == x].index) for x in nodes_ids}
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} features to 1 dimension with probas as y axis'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight, nodes_to_highlight)


def clean_sample(s: str, max_size: int = 100):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    s = s.replace('<br>', '')
    s = s.replace('<br />', '')

    s = s.split(' ')

    if len(s) > max_size:
        s = s[:max_size]
        s[max_size - 1] = s[max_size - 1] + '...'

    if len(s) > 10:
        for i in range(10, min(len(s), max_size), 10):
            s.insert(i, '<br>')

    s = ' '.join(s)

    return s
