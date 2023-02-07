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
import os
import time
import warnings
from typing import Container, List, Tuple, Dict

import pandas as pd
import plotly.graph_objects as go
from PyNomaly import loop
from tensorboard.plugins import projector

from deepchecks.nlp.utils.embeddings import clean_special_chars
from deepchecks.utils import gower_distance

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
import numpy as np
import tensorflow as tf


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
        if value < sum(top_fi.head(index)) * 0.02 or index >= 50:
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
    return all_data, all_embeddings




def run_multivariable_drift(train_dataframe: pd.DataFrame, test_dataframe: pd.DataFrame,
                            numerical_features: List[Hashable], cat_features: List[Hashable], sample_size: int,
                            random_state: int, test_size: float, n_top_columns: int, min_feature_importance: float,
                            max_num_categories_for_display: int, show_categories_by: str,
                            min_meaningful_drift_score: float, with_display: bool,
                            dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES, feature_importance_timeout: int = 120, ):
    """Calculate multivariable drift."""
    train_sample_df = train_dataframe.sample(sample_size, random_state=random_state)[numerical_features + cat_features]
    test_sample_df = test_dataframe.sample(sample_size, random_state=random_state)[numerical_features + cat_features]

    # create new dataset, with label denoting whether sample belongs to test dataset
    domain_class_df = pd.concat([train_sample_df, test_sample_df])
    domain_class_df[cat_features] = RareCategoryEncoder(254).fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_df[cat_features] = OrdinalEncoder().fit_transform(domain_class_df[cat_features].astype(str))
    domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(domain_class_df), domain_class_labels,
                                                        stratify=domain_class_labels, random_state=random_state,
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
    fi, importance_type = calculate_feature_importance_or_none(domain_classifier, domain_test_dataset,
                                                               model_classes=[0, 1], observed_classes=[0, 1],
                                                               task_type=TaskType.BINARY, force_permutation=True,
                                                               permutation_kwargs={'n_repeats': 10,
                                                                                   'random_state': random_state,
                                                                                   'timeout': feature_importance_timeout,
                                                                                   'skip_messages': True})

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
        top_fi = top_fi.loc[top_fi > min_feature_importance]
    else:
        top_fi = None

    if top_fi is not None and len(top_fi):
        score = values_dict['domain_classifier_drift_score']

        displays = [feature_importance_note, build_drift_plot(score), '<h3>Main features contributing to drift</h3>',
                    N_TOP_MESSAGE % n_top_columns,
                    get_drift_plot_sidenote(max_num_categories_for_display, show_categories_by), *(
                display_dist(train_sample_df[feature], test_sample_df[feature], top_fi, cat_features,
                             max_num_categories_for_display, show_categories_by, dataset_names) for feature in
                top_fi.index)]
    else:
        displays = None

    return values_dict, displays



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

    # import random
    # random.seed(random_state)
    # train_sample_df = train_embeddings.sample(sample_size, random_state=random_state)[numerical_features + cat_features]
    # test_sample_df = test_embeddings.sample(sample_size, random_state=random_state)[numerical_features + cat_features]

    # create new dataset, with label denoting whether sample belongs to test dataset
    all_embeddings = pd.concat([train_embeddings, test_embeddings])
    all_embeddings_labels = pd.Series([0] * len(train_embeddings) + [1] * len(test_embeddings), index=all_embeddings.index)
    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(all_embeddings), all_embeddings_labels,
                                                        stratify=all_embeddings_labels, random_state=random_state,
                                                        test_size=test_size)

    # train a model to disguise between train and test samples
    domain_classifier = GradientBoostingClassifier(max_depth=3, subsample=0.8,  min_samples_split=50, n_estimators=30, random_state=random_state)
    domain_classifier.fit(x_train, y_train)

    # calculate feature importance of domain_classifier
    top_fi = pd.Series(domain_classifier.feature_importances_, index=x_train.columns).sort_values(ascending=False).head(30)
    top_fi_embedding = top_fi.head(30).index.values

    domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])
    print('domain_classifier_auc on train is ', roc_auc_score(y_train, domain_classifier.predict_proba(x_train)[:, 1]))
    print('domain_classifier_auc on test is ', domain_classifier_auc)
    drift_score = auc_to_drift_score(domain_classifier_auc)

    values_dict = {'domain_classifier_auc': domain_classifier_auc, 'domain_classifier_drift_score': drift_score,
                   'domain_classifier_feature_importance': top_fi.to_dict() if top_fi is not None else {}, }

    # Sample data before display calculations
    # num_samples_in_display = min(num_samples_in_display, sample_size)
    # train_dataset_for_display = train_dataset.sample(num_samples_in_display, random_state=42)
    # train_embeddings = train_embeddings.loc[train_dataset_for_display.index]
    # train_indexes_to_highlight = [x for x in train_indexes_to_highlight if x in train_dataset_for_display.index]
    # test_dataset_for_display = test_dataset.sample(num_samples_in_display, random_state=42)
    # test_embeddings = test_embeddings.loc[test_dataset_for_display.index]
    # test_indexes_to_highlight = [x for x in test_indexes_to_highlight if x in test_dataset_for_display.index]

    # Calculate display
    # embeddings_for_display = pd.concat([train_embeddings, test_embeddings])
    domain_classifier_probas = domain_classifier.predict_proba(floatify_dataframe(all_embeddings))[:, 1]

    displays = [build_drift_plot(drift_score),
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
                # display_embeddings_node_dist_as_axis(all_embeddings=all_embeddings.iloc[:, top_fi_embedding],
                #                                      all_embeddings_labels=all_embeddings_labels,
                #                                       train_dataset=train_dataset,
                #                                       test_dataset=test_dataset,
                #                                       train_indexes_to_highlight=train_indexes_to_highlight,
                #                                       test_indexes_to_highlight=test_indexes_to_highlight),
                display_embeddings_proba_as_axis(domain_classifier_probas=domain_classifier_probas,
                                                 all_embeddings=all_embeddings.iloc[:, top_fi_embedding],
                                                 train_dataset=train_dataset,
                                                 test_dataset=test_dataset,
                                                 train_indexes_to_highlight=train_indexes_to_highlight,
                                                 test_indexes_to_highlight=test_indexes_to_highlight),
                display_embeddings_proba_highlighted(domain_classifier_probas=domain_classifier_probas,
                                                 all_embeddings=all_embeddings.iloc[:, top_fi_embedding],
                                                 train_dataset=train_dataset,
                                                 test_dataset=test_dataset,
                                                 train_indexes_to_highlight=train_indexes_to_highlight,
                                                 test_indexes_to_highlight=test_indexes_to_highlight)
                ]

    return values_dict, displays


def _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                         train_indexes_to_highlight, indexes_per_node: Dict[str, List[str]] = None):
    import plotly.express as px
    plot_data['dataset'] = ['train_full'] * len(train_dataset.index) + ['test_full'] * len(test_dataset.index)
    plot_data['label'] = train_dataset.label + test_dataset.label
    plot_data['sample'] = train_dataset.text + test_dataset.text
    plot_data['sample'] = plot_data['sample'].apply(clean_sample)

    # Only keep relevant indexes
    plot_data.index = train_dataset.index + test_dataset.index
    if indexes_per_node is None:
        train_to_add = plot_data[plot_data.index.isin(train_indexes_to_highlight)].copy()
        train_to_add['dataset'] = 'classes_unique_to_train'
        test_to_add = plot_data[plot_data.index.isin(test_indexes_to_highlight)].copy()
        test_to_add['dataset'] = 'classes_unique_to_test'
        plot_data = pd.concat([plot_data, train_to_add, test_to_add], ignore_index=True)
    else:
        stuff_to_add = []
        for node_id, indexes in indexes_per_node.items():
            print(f'{node_id} has {len(indexes)} samples')
            to_add = plot_data[plot_data.index.isin(indexes)].copy()
            if len(to_add) == 0:
                continue

            if 'train_full' in to_add['dataset'].tolist():
                print('Percent of train in node is {}'.format(to_add['dataset'].value_counts()['train_full'] / len(to_add)))
            else:
                print('Percent of train in node is 0')
            print('Percent of unique for train in node is {}'.format(len(to_add[to_add.index.isin(train_indexes_to_highlight)]) / len(to_add)))
            print('Percent of unique for test in node is {}'.format(len(to_add[to_add.index.isin(test_indexes_to_highlight)]) / len(to_add)))

            print(to_add['label'].value_counts().to_dict())
            print('*************************************')
            to_add['dataset'] = node_id
            stuff_to_add.append(to_add)
        plot_data = pd.concat([plot_data] + stuff_to_add, ignore_index=True)
    plot_data['dataset'] = plot_data['dataset'].apply(lambda x: '' if '_full' in x else x)
    fig = px.scatter(plot_data, x=1, y=0, color='dataset', hover_data=['label', 'sample'], hover_name='dataset',
                     title=plot_title, height=1000,   color_discrete_sequence=['lightskyblue', 'darkblue', 'violet', 'yellow', 'green'],
                     width=1000, opacity=1)
    fig.update_traces(marker=dict(size=8, line=dict(width=1, color='DarkSlateGrey')), selector=dict(mode='markers'))
    return fig


def _get_tree_dist_values(clf, data):
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    node_indicator = clf.decision_path(data)
    leaf_for_sample = clf.apply(data)
    print(f'Number of leafs is {len(np.unique(leaf_for_sample))}')

    result = np.zeros(len(data))

    for sample_id in range(len(data)):
        # obtain ids of the nodes `sample_id` goes through, i.e., row `sample_id`
        node_index = node_indicator.indices[
                     node_indicator.indptr[sample_id]: node_indicator.indptr[sample_id + 1]]
        for tree_depth, tree_node_id in enumerate(node_index):
            # we arrived to a leaf
            if leaf_for_sample[sample_id] == tree_node_id:
                break

            if data.iloc[sample_id, feature[tree_node_id]] <= threshold[tree_node_id]:
                result[sample_id] = result[sample_id] + 2 ** (clf.max_depth - tree_depth)
            else:
                result[sample_id] = result[sample_id] - 2 ** (clf.max_depth - tree_depth)

    return result


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

    reduced_embeddings = UMAP(init='random', random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])
    # reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(embeddings.loc[:, top_fi_embeddings])

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_title = f'Embeddings in 2D using {method} on top {top_fi_embeddings.shape[0]} features'
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
    top_fi_embeddings = domain_classifier_fi.head(30)
    top_fi_embeddings = top_fi_embeddings.loc[top_fi_embeddings > 0.01].index.values

    train, test, train_labels, test_labels = train_test_split(embeddings.loc[:, top_fi_embeddings], domain_class_labels,
                                                              test_size=0.2, random_state=42)
    min_cluster_size = max(50, int(len(train) * 0.04))
    classifier = DecisionTreeClassifier(max_depth=8, min_samples_leaf=min_cluster_size, random_state=42,
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


def display_embeddings_node_dist_as_axis(all_embeddings, all_embeddings_labels,
                                     train_dataset, test_dataset, train_indexes_to_highlight: List[int],
                                     test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    from sklearn.decomposition import PCA
    from sklearn.tree import DecisionTreeClassifier

    method = 'PCA'

    x_train, x_test, y_train, y_test = train_test_split(floatify_dataframe(all_embeddings), all_embeddings_labels,
                                                        stratify=all_embeddings_labels, test_size=0.3)
    min_cluster_size = max(50, int(len(x_train) * 0.04))
    classifier = DecisionTreeClassifier(max_depth=8, min_samples_leaf=min_cluster_size, random_state=42,
                                        criterion='entropy')
    classifier.fit(x_train, y_train)
    classifier_auc = roc_auc_score(y_test, classifier.predict_proba(x_test)[:, 1])
    print(f'Classifier AUC: {classifier_auc}')

    tree_node_values = y_test.groupby(classifier.apply(x_test)).mean()
    interesting_nodes = tree_node_values[tree_node_values < 0.25].index.tolist() + tree_node_values[
        tree_node_values > 0.75].index.tolist()
    print(f'Number of interesting nodes: {len(interesting_nodes)}')
    node_per_sample = pd.Series(classifier.apply(all_embeddings), index=all_embeddings.index)
    nodes_to_highlight = {x: list(node_per_sample[node_per_sample == x].index) for x in interesting_nodes}

    reduced_embeddings = PCA(n_components=1, random_state=42).fit_transform(all_embeddings)
    plot_data = pd.DataFrame(reduced_embeddings)
    plot_data[1] = _get_tree_dist_values(classifier, all_embeddings)
    plot_title = f'Top {all_embeddings.shape[1]} embeddings to 1 dimension via {method} with tree node distance values as y axis'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight, nodes_to_highlight)


def display_embeddings_proba_as_axis(domain_classifier_probas, all_embeddings,
                                     train_dataset, test_dataset, train_indexes_to_highlight: List[int],
                                     test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good


    from sklearn.decomposition import PCA
    reduced_embeddings = PCA(n_components=1, random_state=42).fit_transform(all_embeddings)

    plot_data = pd.DataFrame(reduced_embeddings)
    plot_data[1] = domain_classifier_probas
    plot_title = f'Embeddings in 1D using PCA on top {all_embeddings.shape[1]} features with domain classifier probas as x axis'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight)

def display_embeddings_proba_highlighted(domain_classifier_probas, all_embeddings,
                                     train_dataset, test_dataset, train_indexes_to_highlight: List[int],
                                     test_indexes_to_highlight: List[int]):
    # TODO: Prototype, go over and make sure code+docs+tests are good

    method = 'UMAP'
    # method = 'PCA'

    if method == 'UMAP':
        from umap import UMAP
        reduced_embeddings = UMAP(init='random', random_state=42, n_components=2).fit_transform(all_embeddings)
    else:
        from sklearn.decomposition import PCA
        reduced_embeddings = PCA(n_components=2, random_state=42).fit_transform(all_embeddings)

    probas_with_index = pd.Series(domain_classifier_probas, index=all_embeddings.index)
    highlight_groups = {}
    if probas_with_index.quantile(.05) < 0.35:
        print(f'Low proba threshold is {probas_with_index.quantile(.05)}')
        highlight_groups['Native to train'] = probas_with_index[(probas_with_index < probas_with_index.quantile(.05))
                                                                & (probas_with_index.index.isin(train_dataset.index))].index.tolist()
    if probas_with_index.quantile(.95) > 0.65:
        print(f'High proba threshold is {probas_with_index.quantile(.95)}')
        highlight_groups['Native to test'] = probas_with_index[(probas_with_index > probas_with_index.quantile(.95))
                                                                 & (probas_with_index.index.isin(test_dataset.index))].index.tolist()


    plot_data = pd.DataFrame(reduced_embeddings)
    plot_title = f'Embeddings in 2D using {method} on top {all_embeddings.shape[1]} features'
    return _draw_plot_from_data(plot_title, plot_data, test_dataset, test_indexes_to_highlight, train_dataset,
                                train_indexes_to_highlight, highlight_groups)


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
    drift_plot = go.Figure(layout=dict(title='Drift Score - Multivariable', xaxis=x_axis, yaxis=y_axis, height=200))

    drift_plot.add_traces(bar_traces)
    return drift_plot


def display_dist(train_column: pd.Series, test_column: pd.Series, fi: pd.Series, cat_features: Container[str],
                 max_num_categories: int, show_categories_by: str, dataset_names: Tuple[str] = DEFAULT_DATASET_NAMES):
    """Create a distribution comparison plot for the given columns."""
    column_name = train_column.name or ''
    column_fi = fi.loc[column_name]
    title = f'Feature: {column_name} - Explains {format_percent(column_fi)} of dataset difference'

    dist_traces, xaxis_layout, yaxis_layout = feature_distribution_traces(train_column.dropna(), test_column.dropna(),
                                                                          column_name,
                                                                          is_categorical=column_name in cat_features,
                                                                          max_num_categories=max_num_categories,
                                                                          show_categories_by=show_categories_by,
                                                                          dataset_names=dataset_names)

    fig = go.Figure()
    fig.add_traces(dist_traces)

    return fig.update_layout(go.Layout(title=title, xaxis=xaxis_layout, yaxis=yaxis_layout,
                                       legend=dict(title='Dataset', yanchor='top', y=0.9, xanchor='left'), height=300))
