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

from typing import Tuple

import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from umap import UMAP

from deepchecks.core.check_utils.multivariate_drift_utils import auc_to_drift_score, build_drift_plot
from deepchecks.nlp import TextData
from deepchecks.nlp.utils.nlp_plot import two_datasets_scatter_plot
from deepchecks.utils.dataframes import floatify_dataframe
from deepchecks.utils.plot import DEFAULT_DATASET_NAMES

SAMPLES_FOR_REDUCTION_FIT = 1000


def run_multivariable_drift_for_embeddings(train_dataset: TextData, test_dataset: TextData,
                                           sample_size: int, random_state: int, test_size: float,
                                           min_meaningful_drift_score: float, num_samples_in_display: int,
                                           dimension_reduction_method: str, with_display: bool,
                                           dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES):
    """Calculate multivariable drift on embeddings."""
    # sample train and test datasets equally
    train_sample = train_dataset.sample(sample_size, random_state=random_state)
    test_sample = test_dataset.sample(sample_size, random_state=random_state)

    train_sample_df = train_sample.embeddings
    test_sample_df = test_sample.embeddings

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_df = pd.concat([train_sample_df, test_sample_df]).reset_index(drop=True)
    domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

    # reduce dimensionality of embeddings if needed.
    # skips if not required ('none') or if number of features is small enough (< 30) in 'auto' mode.
    if not (dimension_reduction_method == 'none' or (
            dimension_reduction_method == 'auto' and domain_class_df.shape[1] < 30)):

        if (dimension_reduction_method == 'auto' and with_display) or dimension_reduction_method == 'umap':
            reducer = UMAP(n_components=10, n_neighbors=5, init='random', random_state=random_state)
        else:  # Faster, but graph will look bad.
            reducer = PCA(n_components=10, random_state=random_state)

        samples_for_reducer = min(SAMPLES_FOR_REDUCTION_FIT, len(domain_class_df))
        reducer.fit(domain_class_df.sample(samples_for_reducer, random_state=random_state))
        domain_class_df = pd.DataFrame(reducer.transform(domain_class_df), index=domain_class_df.index)

        # update train and test samples with new reduced embeddings (used later in display)
        new_embeddings_train = domain_class_df.iloc[:len(train_sample_df)]
        new_embeddings_test = domain_class_df.iloc[len(train_sample_df):]
        train_sample.set_embeddings(new_embeddings_train, verbose=False)
        test_sample.set_embeddings(new_embeddings_test, verbose=False)

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels, random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=2, random_state=random_state)
    domain_classifier.fit(x_train, y_train)

    y_pred = domain_classifier.predict_proba(x_test)[:, 1]
    domain_classifier_auc = roc_auc_score(y_test, y_pred)
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {'domain_classifier_auc': domain_classifier_auc, 'domain_classifier_drift_score': drift_score}

    if with_display and drift_score > min_meaningful_drift_score:
        relevant_index_train = list(x_test[y_test == 0].index)
        relevant_index_test = [x - len(train_sample_df) for x in x_test[y_test == 1].index]

        train_sample = train_sample.copy(rows_to_use=relevant_index_train)
        test_sample = test_sample.copy(rows_to_use=relevant_index_test)

        # Sample data before display calculations
        num_samples_in_display_train = min(int(num_samples_in_display/2), sample_size, len(train_sample))
        train_dataset_for_display = train_sample.sample(num_samples_in_display_train, random_state=random_state)

        num_samples_in_display_test = min(int(num_samples_in_display/2), sample_size, len(test_sample))
        test_dataset_for_display = test_sample.sample(num_samples_in_display_test, random_state=random_state)

        displays = [build_drift_plot(drift_score),
                    display_embeddings(train_dataset=train_dataset_for_display,
                                       test_dataset=test_dataset_for_display,
                                       dataset_names=dataset_names,
                                       random_state=random_state)]
    else:
        displays = None

    return values_dict, displays


def display_embeddings(train_dataset: TextData, test_dataset: TextData, dataset_names: Tuple[str], random_state: int):
    """Display the embeddings with the domain classifier proba as the x-axis and the embeddings as the y-axis."""
    embeddings = pd.concat([train_dataset.embeddings, test_dataset.embeddings])

    reducer = UMAP(n_components=2, n_neighbors=5, init='random', min_dist=1, random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings)

    x_axis_title = 'Reduced Embedding (0)'
    y_axis_title = 'Reduced Embedding (1)'

    plot_data = pd.DataFrame({x_axis_title: reduced_embeddings[:, 0],
                              y_axis_title: reduced_embeddings[:, 1]})
    plot_title = 'Scatter Plot of Embeddings Space (reduced to 2 dimensions)'
    return two_datasets_scatter_plot(plot_title, plot_data, test_dataset, train_dataset, dataset_names)
