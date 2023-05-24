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
import warnings

import numpy as np
import pandas as pd
from numba import NumbaDeprecationWarning
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepchecks.core.check_utils.multivariate_drift_utils import auc_to_drift_score, build_drift_plot
from deepchecks.nlp import TextData
from deepchecks.nlp.utils.nlp_plot import two_datasets_scatter_plot

with warnings.catch_warnings():
    warnings.simplefilter(action='ignore', category=NumbaDeprecationWarning)
    from umap import UMAP

# Max number of samples to use for dimensionality reduction fit (to make calculation faster):
SAMPLES_FOR_REDUCTION_FIT = 1000


def run_multivariable_drift_for_embeddings(train_dataset: TextData, test_dataset: TextData,
                                           sample_size: int, random_state: int, test_size: float,
                                           num_samples_in_display: int, dimension_reduction_method: str,
                                           model_classes: list, with_display: bool):
    """Calculate multivariable drift on embeddings."""
    np.random.seed(random_state)

    # sample train and test datasets equally
    train_sample = train_dataset.sample(sample_size, random_state=random_state)
    test_sample = test_dataset.sample(sample_size, random_state=random_state)

    train_sample_embeddings = train_sample.embeddings
    test_sample_embeddings = test_sample.embeddings

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_array = np.concatenate([train_sample_embeddings, test_sample_embeddings])
    domain_class_labels = pd.Series([0] * len(train_sample_embeddings) + [1] * len(test_sample_embeddings))

    # reduce dimensionality of embeddings if needed.
    # skips if not required ('none') or if number of features is small enough (< 30) in 'auto' mode.
    use_reduction = not (dimension_reduction_method == 'none' or (
            dimension_reduction_method == 'auto' and domain_class_array.shape[1] < 30))
    use_umap = dimension_reduction_method == 'umap' or (dimension_reduction_method == 'auto' and with_display)

    if use_reduction:
        if use_umap:
            reducer = UMAP(n_components=10, n_neighbors=5, init='random',
                           random_state=np.random.RandomState(random_state))
        else:  # Faster, but graph will look bad.
            reducer = PCA(n_components=10, random_state=random_state)

        samples_for_reducer = min(SAMPLES_FOR_REDUCTION_FIT, len(domain_class_array))
        samples = np.random.choice(len(domain_class_array), samples_for_reducer, replace=False)
        reducer.fit(domain_class_array[samples])
        domain_class_array = reducer.transform(domain_class_array)

        # update train and test samples with new reduced embeddings (used later in display)
        new_embeddings_train = domain_class_array[:len(train_sample_embeddings)]
        new_embeddings_test = domain_class_array[len(train_sample_embeddings):]
        train_sample.set_embeddings(new_embeddings_train, verbose=False)
        test_sample.set_embeddings(new_embeddings_test, verbose=False)

    x_train, x_test, y_train, y_test = train_test_split(domain_class_array, domain_class_labels,
                                                        stratify=domain_class_labels, random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=2, random_state=random_state)
    domain_classifier.fit(x_train, y_train)

    y_pred = domain_classifier.predict_proba(x_test)[:, 1]
    domain_classifier_auc = roc_auc_score(y_test, y_pred)
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {'domain_classifier_auc': domain_classifier_auc, 'domain_classifier_drift_score': drift_score}

    if with_display:
        relevant_index_train = list(y_test[y_test == 0].index)
        relevant_index_test = [x - len(train_sample_embeddings) for x in y_test[y_test == 1].index]

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
                                       random_state=random_state,
                                       model_classes=model_classes)]
    else:
        displays = None

    return values_dict, displays


def display_embeddings(train_dataset: TextData, test_dataset: TextData, random_state: int, model_classes: list):
    """Display the embeddings with the domain classifier proba as the x-axis and the embeddings as the y-axis."""
    embeddings = np.concatenate([train_dataset.embeddings, test_dataset.embeddings])

    reducer = UMAP(n_components=2, n_neighbors=5, init='random', min_dist=1, random_state=random_state)
    reduced_embeddings = reducer.fit_transform(embeddings)

    x_axis_title = 'Reduced Embedding (0)'
    y_axis_title = 'Reduced Embedding (1)'

    plot_data = pd.DataFrame({x_axis_title: reduced_embeddings[:, 0],
                              y_axis_title: reduced_embeddings[:, 1]})
    plot_title = 'Scatter Plot of Embeddings Space (reduced to 2 dimensions)'
    return two_datasets_scatter_plot(plot_title=plot_title, plot_data=plot_data, train_dataset=train_dataset,
                                     test_dataset=test_dataset, model_classes=model_classes)
