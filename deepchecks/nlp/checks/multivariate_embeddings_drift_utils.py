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
"""Module containing common EmbeddingsDrift Check (domain classifier drift) utils."""

import numpy as np
from typing import List, Tuple

import pandas as pd
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepchecks.core.check_utils.multivariate_drift_utils import auc_to_drift_score, build_drift_plot
from deepchecks.utils.dataframes import floatify_dataframe
from deepchecks.utils.distribution.drift import get_drift_plot_sidenote
from deepchecks.utils.distribution.plot import drift_score_bar_traces, feature_distribution_traces
from deepchecks.utils.distribution.rare_category_encoder import RareCategoryEncoder
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES, colors
from deepchecks.utils.strings import format_percent
from deepchecks.utils.typing import Hashable
from sklearn.decomposition import PCA
from umap import UMAP

SAMPLES_FOR_REDUCTION_FIT = 1000


def run_multivariable_drift_for_embeddings(train_embeddings: pd.DataFrame, test_embeddings: pd.DataFrame, train_dataset,
                                           test_dataset,
                                           numerical_features: List[Hashable], cat_features: List[Hashable],
                                           sample_size: int, random_state: int, test_size: float, n_top_columns: int,
                                           min_feature_importance: float, min_meaningful_drift_score: float,
                                           num_samples_in_display: int, dimension_reduction_method: str,
                                           with_display: bool, dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES):
    """Calculate multivariable drift."""

    train_sample = train_dataset.sample(sample_size, random_state=random_state)
    test_sample = test_dataset.sample(sample_size, random_state=random_state)

    train_sample_df = train_sample.embeddings
    test_sample_df = test_sample.embeddings

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_df = pd.concat([train_sample_df, test_sample_df]).reset_index(drop=True)
    domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

    if not (dimension_reduction_method == 'none' or (
            dimension_reduction_method == 'auto' and domain_class_df.shape[1] < 30)):
        if (dimension_reduction_method == 'auto' and with_display) or dimension_reduction_method == 'umap':
            reducer = UMAP(n_components=10, n_neighbors=5, init='random', random_state=random_state)
        else:
            reducer = PCA(n_components=10, random_state=random_state)

        samples_for_reducer = min(SAMPLES_FOR_REDUCTION_FIT, len(domain_class_df))
        reducer.fit(domain_class_df.sample(samples_for_reducer, random_state=random_state))
        domain_class_df = pd.DataFrame(reducer.transform(domain_class_df), index=domain_class_df.index)

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels, random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=2, random_state=random_state)
    domain_classifier.fit(x_train, y_train)
    
    # calculate feature importance of domain_classifier, containing the information which features separate
    # the dataset best.

    y_pred = domain_classifier.predict_proba(x_test)[:, 1]

    domain_classifier_auc = roc_auc_score(y_test, y_pred)
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {'domain_classifier_auc': domain_classifier_auc, 'domain_classifier_drift_score': drift_score}

    if with_display and drift_score > min_meaningful_drift_score:
        fi = pd.Series(domain_classifier.feature_importances_, index=x_train.columns).sort_values(ascending=False)
        top_fi = fi.head(n_top_columns)
        top_fi_over_threshold = top_fi[top_fi >= min_feature_importance]
        if top_fi_over_threshold.shape[0] >= 2:
            top_fi = top_fi_over_threshold

        train_embeddings = pd.DataFrame(reducer.transform(train_sample_df), index=train_sample_df.index)
        test_embeddings = pd.DataFrame(reducer.transform(test_sample_df), index=test_sample_df.index)

        # train_embeddings = pd.DataFrame(reducer.transform(train_embeddings), index=train_embeddings.index)
        # test_embeddings = pd.DataFrame(reducer.transform(test_embeddings), index=test_embeddings.index)

        # train_embeddings = x_test[y_test == 0]
        # test_embeddings = x_test[y_test == 1]

        # TEMPORARY
        train_sample.set_embeddings(train_embeddings)
        test_sample.set_embeddings(test_embeddings)


        # Sample data before display calculations
        num_samples_in_display_train = min(num_samples_in_display, sample_size, len(train_embeddings))

        # samples_for_display = np.random.choice(range(len(train_embeddings)), size=num_samples_in_display_train,
        #                                        replace=False).tolist()

        train_dataset_for_display = train_sample.sample(num_samples_in_display_train, random_state=random_state)
        train_embeddings = train_dataset_for_display.embeddings

        num_samples_in_display_test = min(num_samples_in_display, sample_size, len(test_embeddings))
        # samples_for_display = np.random.choice(range(len(test_embeddings)), size=num_samples_in_display_test,
        #                                        replace=False).tolist()
        test_dataset_for_display = test_sample.sample(num_samples_in_display_test, random_state=random_state)
        test_embeddings = test_dataset_for_display.embeddings

        # Calculate display
        embeddings_for_display = pd.concat([train_embeddings, test_embeddings]).reset_index(drop=True)
        domain_classifier_probas = domain_classifier.predict_proba(floatify_dataframe(embeddings_for_display))[:, 1]

        displays = [build_drift_plot(drift_score),
                    display_embeddings_proba_as_axis(domain_classifier_probas=domain_classifier_probas,
                                                     train_embeddings=train_embeddings, test_embeddings=test_embeddings,
                                                     top_fi_embeddings=top_fi, train_dataset=train_dataset_for_display,
                                                     test_dataset=test_dataset_for_display,
                                                     dataset_names=dataset_names)
                    ]
    else:
        displays = None

    return values_dict, displays


def _draw_plot_from_data(plot_title, plot_data, test_dataset, train_dataset,
                         dataset_names):
    import plotly.express as px
    plot_data['dataset'] = [dataset_names[0]] * len(train_dataset) + [dataset_names[1]] * len(test_dataset)
    if train_dataset.has_label():
        plot_data['label'] = np.concatenate([train_dataset.label_for_display, test_dataset.label_for_display])
    else:
        plot_data['label'] = None
    plot_data['sample'] = np.concatenate([train_dataset.text, test_dataset.text])
    plot_data['sample'] = plot_data['sample'].apply(clean_sample)

    fig = px.scatter(plot_data, x=1, y=0, color='dataset', color_discrete_map=colors, hover_data=['label', 'sample'],
                     hover_name='dataset', title=plot_title, height=600, width=1000, opacity=0.4)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig


def display_embeddings_proba_as_axis(domain_classifier_probas, train_embeddings, test_embeddings, top_fi_embeddings,
                                     train_dataset, test_dataset, dataset_names):
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
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, train_dataset, dataset_names)


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
