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
"""Module containing common MultivariateDrift Check (domain classifier drift) utils."""
import warnings
from typing import Container, List, Tuple

import pandas as pd
import plotly.graph_objects as go

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


def run_multivariable_drift_for_embeddings(
        train_embeddings: pd.DataFrame, test_embeddings: pd.DataFrame,
        train_dataset, test_dataset,  # Type TextData but I don't want to import
        numerical_features: List[Hashable], cat_features: List[Hashable], sample_size: int,
        random_state: int, test_size: float, n_top_columns: int, min_feature_importance: float,
        min_meaningful_drift_score: float, num_samples_in_display: int,
        with_display: bool,
        dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES,
        train_indexes_to_highlight: List[int] = [],
        test_indexes_to_highlight: List[int] = [],
):
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

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels,
                                                        random_state=random_state,
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
        # Sample data before display calculations
        num_samples_in_display = min(num_samples_in_display, sample_size)
        train_dataset = train_dataset.sample(num_samples_in_display)
        train_embeddings = train_embeddings[train_embeddings.index.isin(train_dataset.index)]
        train_indexes_to_highlight = [x for x in train_indexes_to_highlight if x in train_dataset.index]
        test_dataset = test_dataset.sample(num_samples_in_display)
        test_embeddings = test_embeddings[test_embeddings.index.isin(test_dataset.index)]
        test_indexes_to_highlight = [x for x in test_indexes_to_highlight if x in test_dataset.index]

        # Calculate display
        embeddings_for_display = pd.concat([train_embeddings, test_embeddings])
        domain_classifier_probas = domain_classifier.predict_proba(floatify_dataframe(embeddings_for_display))[:, 1]

        displays = [
            feature_importance_note,
            build_drift_plot(score),
            display_embeddings(train_embeddings=train_embeddings,
                               test_embeddings=test_embeddings,
                               top_fi_embeddings=top_fi, train_dataset=train_dataset,
                               test_dataset=test_dataset, train_indexes_to_highlight=train_indexes_to_highlight,
                               test_indexes_to_highlight=test_indexes_to_highlight),
            display_embeddings_with_clusters_by_nodes(
                train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_indexes_to_highlight=train_indexes_to_highlight,
                               test_indexes_to_highlight=test_indexes_to_highlight),
            display_embeddings_with_domain_classifier(
                domain_classifier_probas=domain_classifier_probas, train_embeddings=train_embeddings,
                test_embeddings=test_embeddings,
                top_fi_embeddings=top_fi, train_dataset=train_dataset,
                test_dataset=test_dataset,
                train_indexes_to_highlight=train_indexes_to_highlight,
                               test_indexes_to_highlight=test_indexes_to_highlight)
        ]
    else:
        displays = None

    return values_dict, displays


def _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight,
                         train_dataset, train_indexes_to_highlight):
    import plotly.express as px
    plot_data['dataset'] = ['train_full'] * len(train_dataset.index) + ['test_full'] * len(test_dataset.index)
    plot_data['label'] = train_dataset.label + test_dataset.label
    plot_data['sample'] = train_dataset.text + test_dataset.text
    plot_data['sample'] = plot_data['sample'].apply(clean_sample)

    # Only keep relevant indexes
    plot_data.index = train_dataset.index + test_dataset.index
    train_to_add = plot_data[plot_data.index.isin(train_indexes_to_highlight)]
    train_to_add.loc['dataset'] = 'train_highlighted'
    test_to_add = plot_data[plot_data.index.isin(test_indexes_to_highlight)]
    test_to_add.loc['dataset'] = 'test_highlighted'
    plot_data = pd.concat([plot_data, train_to_add, test_to_add], ignore_index=True)
    fig = px.scatter(plot_data, x=1, y=0, color='dataset', hover_data=['label', 'sample'], hover_name='dataset',
                     title=plot_title,
                     height=1000, width=1000, opacity=0.4)
    fig.update_traces(marker=dict(size=8,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))
    return fig

def display_embeddings(train_embeddings, test_embeddings, top_fi_embeddings, train_dataset, test_dataset,
                       train_indexes_to_highlight: List[int], test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    if top_fi_embeddings.shape[0] == 1:
        return ''

    from umap import UMAP
    # from sklearn.decomposition import PCA

    method = 'UMAP'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    top_fi_embeddings = top_fi_embeddings.index.values

    reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    # reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_title = 'regular'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight,
                                train_dataset, train_indexes_to_highlight)


def display_embeddings_with_clusters_by_nodes(train_embeddings, test_embeddings, train_dataset, test_dataset,
                                              train_indexes_to_highlight: List[int], test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    from umap import UMAP
    # from sklearn.decomposition import PCA

    method = 'UMAP'
    # method = 'PCA'

    embeddings = pd.concat([train_embeddings, test_embeddings])

    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=42)
    domain_class_labels = pd.Series([0] * len(train_embeddings) + [1] * len(test_embeddings))

    classifier.fit(embeddings, domain_class_labels)

    fi = pd.Series(classifier.feature_importances_, index=train_embeddings.columns)

    fi = fi.sort_values(ascending=False) if fi is not None else None

    top_fi_embeddings = fi.head(20)
    top_fi_embeddings = top_fi_embeddings.loc[top_fi_embeddings > 0.01]

    top_fi_embeddings = top_fi_embeddings.index.values

    domain_classifier_nodes = classifier.apply(embeddings)

    # reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    reduced_embeddings = UMAP(n_components=2, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings], y=domain_classifier_nodes)

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_title = 'node indexes'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight,
                                train_dataset, train_indexes_to_highlight)


def display_embeddings_with_domain_classifier(domain_classifier_probas, train_embeddings, test_embeddings,
                                              top_fi_embeddings, train_dataset, test_dataset,
                                              train_indexes_to_highlight: List[int], test_indexes_to_highlight: List[int]):
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
    plot_title = 'domain classifier proba'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight,
                                train_dataset, train_indexes_to_highlight)


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
