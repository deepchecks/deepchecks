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
"""Utils module for calculating embeddings display checks."""

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from PyNomaly import loop
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from deepchecks.core.check_utils.multivariate_drift_utils import auc_to_drift_score, build_drift_plot
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.nlp.utils.embeddings_calculator import _clean_special_chars
from deepchecks.utils.dataframes import floatify_dataframe


def create_performance_files(text: pd.Series, embeddings: np.ndarray, proba: np.ndarray, y_true: np.ndarray,
                             labels: List[str], path: str, sample_size: int = 10000, verbose: bool = False,
                             indexes_to_highlight: Dict[str, List[int]] = None):
    if list(sorted(labels)) != list(labels):
        raise DeepchecksValueError('Labels must be sorted in an alphanumeric order')
    if not os.path.exists(path):
        os.makedirs(path)

    loss_per_sample = [log_loss([y_true], [y_proba], labels=sorted(labels)) for y_true, y_proba in zip(y_true, proba)]

    text_dataframe = pd.DataFrame(text)
    text_dataframe.columns = ['text']
    text_dataframe['loss per sample'] = loss_per_sample
    text_dataframe['label'] = y_true
    text_dataframe['prediction'] = [labels[np.argmax(x)] for x in proba]
    text_dataframe['model uncertainty'] = [(1 - np.max(x)) for x in proba]
    text_dataframe['correct prediction'] = [x == y for x, y in
                                            zip(text_dataframe['label'], text_dataframe['prediction'])]

    if indexes_to_highlight is not None:
        text_dataframe['highlight criteria'] = [_select_highlight_value(x, indexes_to_highlight) for x in
                                                text_dataframe.index]
    text_dataframe['text'] = text_dataframe['text'].apply(_clean_special_chars)
    text_dataframe = text_dataframe[['text'] + [col for col in text_dataframe.columns if col != 'text']]

    # save values for display
    text_dataframe.to_csv(os.path.join(path, 'performance_metadata.tsv'), sep='\t')
    checkpoint = tf.train.Checkpoint(
        performance_embedding=tf.Variable(pd.DataFrame(embeddings, index=text_dataframe.index)))
    checkpoint.save(os.path.join(path, 'performance_embedding.ckpt'))


def create_outlier_files(text: pd.Series, embeddings: np.ndarray, path: str, nearest_neighbors_percent: float = 0.01,
                         extent_parameter: int = 3, sample_size: int = 10000, verbose: bool = False,
                         indexes_to_highlight: Dict[str, List[int]] = None):
    if not os.path.exists(path):
        os.makedirs(path)
    start_time = time.time()
    text_dataframe = pd.DataFrame(text)
    text_dataframe.columns = ['text']
    embeddings_df = pd.DataFrame(embeddings, index=text_dataframe.index)
    num_neighbors = max(int(nearest_neighbors_percent * len(embeddings_df)), 10)

    # from umap import UMAP
    # embeddings_df = UMAP(init='random', random_state=42, n_components=3).fit_transform(embeddings_df)

    # Calculate outlier probability score using loop algorithm.
    m = loop.LocalOutlierProbability(embeddings_df, extent=extent_parameter, n_neighbors=num_neighbors).fit()
    prob_vector = np.asarray(m.local_outlier_probabilities, dtype=float)
    text_dataframe['outlier probability'] = prob_vector

    if indexes_to_highlight is not None:
        text_dataframe['highlight criteria'] = [_select_highlight_value(x, indexes_to_highlight) for x in
                                                text_dataframe.index]
    # make text be the first column
    text_dataframe['text'] = text_dataframe['text'].apply(_clean_special_chars)
    text_dataframe = text_dataframe[['text'] + [col for col in text_dataframe.columns if col != 'text']]
    if verbose:
        print(f'finished calculating outlier probability score after {time.time() - start_time} seconds')
        print(f'Avg outlier score is {np.mean(prob_vector)}')
        if indexes_to_highlight is not None:
            print(f'Avg outlier score in real outlier samples is '
                  f'{np.mean(prob_vector[text_dataframe.index.isin(indexes_to_highlight["real outlier samples"])])}')

    # save to file
    text_dataframe.to_csv(os.path.join(path, 'outlier_metadata.tsv'), sep='\t')

    # save_embeddings_to_file
    checkpoint = tf.train.Checkpoint(outlier_embedding=tf.Variable(embeddings_df))
    checkpoint.save(os.path.join(path, 'outlier_embedding.ckpt'))


def _select_highlight_value(index, indexes_to_highlight: Dict[str, List[int]]):
    for key, value in indexes_to_highlight.items():
        if index in value:
            return key
    return 'other'


def create_drift_files(train_text: pd.Series, test_text: pd.Series, train_embeddings: np.ndarray,
                       test_embeddings: np.ndarray, path: str, sample_size: int = 2500, verbose: bool = False,
                       additional_data: pd.DataFrame = None):
    if not os.path.exists(path):
        os.makedirs(path)

    train_dataframe, test_dataframe = pd.DataFrame(train_text), pd.DataFrame(test_text)
    train_dataframe.columns, test_dataframe.columns = ['text'], ['text']
    train_embeddings = pd.DataFrame(train_embeddings, index=train_dataframe.index)
    test_embeddings = pd.DataFrame(test_embeddings, index=test_dataframe.index)

    sample_size = min(sample_size, len(train_dataframe), len(test_dataframe))
    if len(train_dataframe) > sample_size:
        train_dataframe = train_dataframe.sample(sample_size)
        train_embeddings = train_embeddings.loc[train_dataframe.index]
    if len(test_dataframe) > sample_size:
        test_dataframe = test_dataframe.sample(sample_size)
        test_embeddings = test_embeddings.loc[test_dataframe.index]

    all_embeddings = pd.concat([train_embeddings, test_embeddings], ignore_index=True)
    all_embeddings_labels = [0] * len(train_embeddings) + [1] * len(test_embeddings)
    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(all_embeddings), all_embeddings_labels,
                                                        stratify=all_embeddings_labels, test_size=0.2)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=3, subsample=0.8,
                                                   min_samples_split=50, n_estimators=30, random_state=42)
    domain_classifier.fit(x_train, y_train)

    domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])
    drift_score = auc_to_drift_score(domain_classifier_auc)
    if verbose:
        print('domain_classifier_auc on train is ',
              roc_auc_score(y_train, domain_classifier.predict_proba(x_train)[:, 1]))
        print('domain_classifier_auc on test is ', domain_classifier_auc)
        print(f'Domain classifier drift score is {drift_score:.2f}')
    train_dataframe['domain classifier proba'] = domain_classifier.predict_proba(train_embeddings)[:, 1]
    test_dataframe['domain classifier proba'] = domain_classifier.predict_proba(test_embeddings)[:, 1]

    # create metadata file for display
    train_dataframe['sample origin'] = 'train'
    test_dataframe['sample origin'] = 'test'
    all_data = pd.concat([train_dataframe, test_dataframe])
    if additional_data is not None:
        additional_data = additional_data.loc[all_data.index]
        all_data = all_data.join(additional_data)

    all_data['domain classifier proba'] = all_data['domain classifier proba'].round(2)
    all_data['text'] = all_data['text'].apply(_clean_special_chars)
    all_data = all_data[['text'] + [col for col in all_data.columns if col != 'text']]

    # save data for display
    all_data.to_csv(os.path.join(path, 'drift_metadata.tsv'), sep='\t')
    checkpoint = tf.train.Checkpoint(drift_embeddings=tf.Variable(all_embeddings))
    checkpoint.save(os.path.join(path, 'drift_embeddings.ckpt'))

    if verbose:
        return all_data, all_embeddings
    else:
        return build_drift_plot(drift_score)
