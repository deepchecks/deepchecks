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
"""Utils module for calculating embeddings display checks."""

import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
import tensorflow as tf
from PyNomaly import loop
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from deepchecks.core.check_utils.multivariate_drift_utils import auc_to_drift_score, build_drift_plot
from deepchecks.nlp.utils.embeddings_calculator import clean_special_chars
from deepchecks.utils.dataframes import floatify_dataframe


def create_outlier_display(text: pd.Series, embeddings: np.ndarray,
                             path: str, nearest_neighbors_percent: float = 0.01, extent_parameter: int = 3,
                             sample_size: int = 10000, verbose: bool = False,
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


    # Calculate outlier probability score using IsoForest algorithm.
    # from sklearn.ensemble import IsolationForest
    # clf = IsolationForest()
    # clf.fit(embeddings_df)
    # prob_vector = np.asarray(clf.decision_function(embeddings_df), dtype=float)
    # text_dataframe['outlier probability'] = prob_vector

    if indexes_to_highlight is not None:
        text_dataframe['highlight criteria'] = [_can_highlight_value(x, indexes_to_highlight) for x in text_dataframe.index]
    # make text be the first column
    text_dataframe['text'] = text_dataframe['text'].apply(clean_special_chars)
    text_dataframe = text_dataframe[['text'] + [col for col in text_dataframe.columns if col != 'text']]
    if verbose:
        print('finished calculating outlier probability score after {} seconds'.format(time.time() - start_time))
        print(f'Avg outlier score is {np.mean(prob_vector)}')
        if indexes_to_highlight is not None:
            print(f'Avg outlier score in real outlier samples is {np.mean(prob_vector[text_dataframe.index.isin(indexes_to_highlight["real outlier samples"])])}')
            # true_outlier_found = text_dataframe[(text_dataframe['highlight criteria'] == 'real outlier samples') & (text_dataframe['is outlier'] == 'outlier')]
            # print(f'Percent of real outlier in detected outliers is {len(true_outlier_found) / len(text_dataframe[text_dataframe["is outlier"] == "outlier"])}')

    # save to file
    text_dataframe.to_csv(os.path.join(path, "outlier_metadata.tsv"), sep="\t")

    # save_embeddings_to_file
    # if not os.path.exists(os.path.join(path, "embedding.ckpt-1.index")):
    checkpoint = tf.train.Checkpoint(outlier_embedding=tf.Variable(embeddings_df))
    checkpoint.save(os.path.join(path, "outlier_embedding.ckpt"))

def _can_highlight_value(index, indexes_to_highlight: Dict[str, List[int]]):
    for key, value in indexes_to_highlight.items():
        if index in value:
            return key
    return "other"

def create_embedding_display(train_text: pd.Series, test_text: pd.Series,
                             train_embeddings: np.ndarray, test_embeddings: np.ndarray,
                             path: str, sample_size: int = 10000, verbose: bool = False,
                             indexes_to_highlight: Dict[str, List[int]] = None):
    if not os.path.exists(path):
        os.makedirs(path)

    train_dataframe, test_dataframe = pd.DataFrame(train_text), pd.DataFrame(test_text)
    train_dataframe.columns, test_dataframe.columns = ['text'], ['text']
    train_embeddings = pd.DataFrame(train_embeddings, index = train_dataframe.index)
    test_embeddings = pd.DataFrame(test_embeddings, index = test_dataframe.index)

    all_embeddings = pd.concat([train_embeddings, test_embeddings], ignore_index=True)
    all_embeddings_labels = [0] * len(train_embeddings) + [1] * len(test_embeddings)
    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(all_embeddings), all_embeddings_labels,
                                                        stratify=all_embeddings_labels, test_size=0.2)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=3, subsample=0.8, min_samples_split=50, n_estimators=30)
    domain_classifier.fit(x_train, y_train)

    # calculate feature importance of domain_classifier
    top_fi = pd.Series(domain_classifier.feature_importances_, index=x_train.columns)\
        .sort_values(ascending=False)
    for index, value in enumerate(top_fi):
        if value < sum(top_fi.head(index)) * 0.01 or index >= 50:
            top_fi = top_fi.head(index)
            break


    domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])
    drift_score = auc_to_drift_score(domain_classifier_auc)
    if verbose:
        print(f'Using {len(top_fi)} embeddings features for display')
        print('domain_classifier_auc on train is ', roc_auc_score(y_train, domain_classifier.predict_proba(x_train)[:, 1]))
        print('domain_classifier_auc on test is ', domain_classifier_auc)
    print(f'Domain classifier drift score is {drift_score:.2f}')
    train_dataframe['domain classifier proba'] = domain_classifier.predict_proba(train_embeddings)[:, 1]
    test_dataframe['domain classifier proba'] = domain_classifier.predict_proba(test_embeddings)[:, 1]


    # create metadata file for display
    train_dataframe['sample origin'] = 'train'
    test_dataframe['sample origin'] = 'test'
    all_data = pd.concat([train_dataframe, test_dataframe])

    lower_limit, upper_limit = all_data['domain classifier proba'].quantile(.05), all_data['domain classifier proba'].quantile(.95)
    all_data['drifted_group'] = ['native to train' if x < lower_limit else 'native to test' if x > upper_limit
        else 'common to both' for x in all_data['domain classifier proba']]

    all_data['domain classifier proba'] = all_data['domain classifier proba'].round(2)
    if indexes_to_highlight is not None:
        all_data['highlight criteria'] = [_can_highlight_value(x, indexes_to_highlight) for x in all_data.index]
    # make text be the first column
    all_data['text'] = all_data['text'].apply(clean_special_chars)
    all_data = all_data[['text'] + [col for col in all_data.columns if col != 'text']]

    #sample and save to file
    all_data.to_csv(os.path.join(path,"drift_metadata.tsv"), sep="\t")

    # save_embeddings_to_file
    # if not os.path.exists(os.path.join(path, "embedding.ckpt-1.index")):
    checkpoint = tf.train.Checkpoint(drift_embeddings=tf.Variable(all_embeddings.iloc[:, top_fi.index.values]))
    # checkpoint = tf.train.Checkpoint(drift_embeddings=tf.Variable(all_embeddings))
    checkpoint.save(os.path.join(path, "drift_embeddings.ckpt"))
    if verbose:
        return all_data, all_embeddings
    else:
        return build_drift_plot(drift_score)
