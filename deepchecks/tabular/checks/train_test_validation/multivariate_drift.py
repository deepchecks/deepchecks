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
"""Module contains the domain classifier drift check."""

from collections import OrderedDict
from copy import copy

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTENC
from merge_args import merge_args
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.check_utils.multivariate_drift_utils import (basic_preprocessing_for_domain_classification,
                                                                  get_domain_classifier_hq, run_multivariable_drift)
from deepchecks.core.fix_classes import TrainTestCheckFixMixin, FixResult
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.strings import format_number
from deepchecks.tabular._shared_docs import docstrings


__all__ = ['MultivariateDrift']


class MultivariateDrift(TrainTestCheck, TrainTestCheckFixMixin):
    """
    Calculate drift between the entire train and test datasets using a model trained to distinguish between them.

    Check fits a new model to distinguish between train and test datasets, called a Domain Classifier.
    Once the Domain Classifier is fitted the check calculates the feature importance for the domain classifier
    model. The result of the check is based on the AUC of the domain classifier model, and the check displays
    the change in distribution between train and test for the top features according to the
    calculated feature importance.

    Parameters
    ----------
    n_top_columns : int , default: 3
        Amount of columns to show ordered by domain classifier feature importance. This limit is used together
        (AND) with min_feature_importance, so less than n_top_columns features can be displayed.
    min_feature_importance : float , default: 0.05
        Minimum feature importance to show in the check display. Feature importance
        sums to 1, so for example the default value of 0.05 means that all features with importance contributing
        less than 5% to the predictive power of the Domain Classifier won't be displayed. This limit is used
        together (AND) with n_top_columns, so features more important than min_feature_importance can be
        hidden.
    max_num_categories_for_display: int, default: 10
        Max number of categories to show in plot.
    show_categories_by: str, default: 'largest_difference'
        Specify which categories to show for categorical features' graphs, as the number of shown categories is limited
        by max_num_categories_for_display. Possible values:
        - 'train_largest': Show the largest train categories.
        - 'test_largest': Show the largest test categories.
        - 'largest_difference': Show the largest difference between categories.
    sample_size : int , default: 10_000
        Max number of rows to use from each dataset for the training and evaluation of the domain classifier.
    random_state : int , default: 42
        Random seed for the check.
    test_size : float , default: 0.3
        Fraction of the combined datasets to use for the evaluation of the domain classifier.
    min_meaningful_drift_score : float , default 0.05
        Minimum drift score for displaying drift in check. Under that score, check will display "nothing found".
    """

    def __init__(
            self,
            n_top_columns: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            n_samples: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3,
            min_meaningful_drift_score: float = 0.05,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.n_top_columns = n_top_columns
        self.min_feature_importance = min_feature_importance
        self.max_num_categories_for_display = max_num_categories_for_display
        self.show_categories_by = show_categories_by
        self.n_samples = n_samples
        self.random_state = random_state
        self.test_size = test_size
        self.min_meaningful_drift_score = min_meaningful_drift_score

    def run_logic(self, context: Context) -> CheckResult:
        """Run check.

        Returns
        -------
        CheckResult
            value: dictionary containing the domain classifier auc and a dict of column name to its feature
            importance as calculated for the domain classifier model.
            display: distribution graph for each column for the columns most explaining the dataset difference,
            comparing the train and test distributions.

        Raises
        ------
        DeepchecksValueError
            If the object is not a Dataset or DataFrame instance
        """
        train_dataset = context.train
        test_dataset = context.test
        cat_features = train_dataset.cat_features
        numerical_features = train_dataset.numerical_features

        if self.n_samples is None:
            sample_size = min(train_dataset.n_samples, test_dataset.n_samples)
        else:
            sample_size = min(train_dataset.n_samples, test_dataset.n_samples, self.n_samples)

        headnote = """
        <span>
        The shown features are the features that are most important for the domain classifier - the
        domain_classifier trained to distinguish between the train and test datasets.<br>
        </span>
        """

        values_dict, displays = run_multivariable_drift(
            train_dataframe=train_dataset.features_columns,
            test_dataframe=test_dataset.features_columns,
            numerical_features=numerical_features,
            cat_features=cat_features,
            sample_size=sample_size, random_state=self.random_state,
            test_size=self.test_size, n_top_columns=self.n_top_columns,
            min_feature_importance=self.min_feature_importance,
            max_num_categories_for_display=self.max_num_categories_for_display,
            show_categories_by=self.show_categories_by,
            min_meaningful_drift_score=self.min_meaningful_drift_score,
            with_display=context.with_display,
            dataset_names=(train_dataset.name, test_dataset.name),
            feature_importance_timeout=context.feature_importance_timeout,
        )

        if displays:
            displays.insert(0, headnote)

        return CheckResult(value=values_dict, display=displays, header='Multivariate Drift')

    def add_condition_overall_drift_value_less_than(self, max_drift_value: float = 0.25):
        """Add condition.

        Overall drift score, calculated as (2 * AUC - 1) for the AUC of the dataset discriminator model, is less
        than the specified value. This value is used as it scales the AUC value to the range [0, 1], where 0 indicates
        a random model (and no drift) and 1 indicates a perfect model (and completely distinguishable datasets).

        Parameters
        ----------
        max_drift_value : float , default: 0.25
            Maximal drift value allowed (value 0 and above)
        """

        def condition(result: dict):
            drift_score = result['domain_classifier_drift_score']
            details = f'Found drift value of: {format_number(drift_score)}, corresponding to a domain classifier ' \
                      f'AUC of: {format_number(result["domain_classifier_auc"])}'
            category = ConditionCategory.PASS if drift_score < max_drift_value else ConditionCategory.FAIL
            return ConditionResult(category, details)

        return self.add_condition(f'Drift value is less than {format_number(max_drift_value)}',
                                  condition)

    @docstrings
    @merge_args(TrainTestCheck.run)
    def fix(self, *args, check_result: CheckResult = None, drop_train: bool = True,
                  oversample_train: bool = True, move_from_test: bool = False, max_move_from_test_to_train: float = 0.3,
                  use_smote: bool = True, max_drop_train: float = 0.4, **kwargs) -> FixResult:
        """Run fix.

        Parameters
        ----------
        {additional_context_params:2*indent}
        check_result : CheckResult
            CheckResult object.
        drop_train : bool, default: True
            Whether to drop samples from the train dataset.
        oversample_train : bool, default: True
            Whether to oversample the train dataset.
        move_from_test : bool, default: True
            Whether to move samples from the test dataset to the train dataset.
        max_move_from_test_to_train : float, default: 0.2
            The maximum percentage of samples to move from the test dataset to the train dataset.
        use_smote : bool, default: True
            Whether to use SMOTE to oversample the train dataset.
        max_drop_train : float, default: 0.4
            The maximum percentage of samples to drop from the train dataset.
        """
        context = self.get_context(*args, **kwargs)

        #TODO: Not production ready, go over all parameters etc.

        # if check_result is None:
        #     check_result = self.run_logic(context)

        np.random.seed(self.random_state)
        revert_index = False
        significant_threshold_test = 0.75
        significant_threshold_train = 0.25
        k_neighbors = 5  # For SMOTENC
        new_to_org_index_dict_train = {}
        new_to_org_index_dict_test = {}

        train, test = context.train, context.test
        cat_features = train.cat_features

        max_n_samples_to_move = int(max_move_from_test_to_train * test.n_samples)
        max_n_samples_to_keep = int((1 - max_drop_train) * train.n_samples)
        max_n_samples_to_drop = int(max_drop_train * train.n_samples)

        train_data = train.features_columns
        test_data = test.features_columns

        train_label = train.label_col if train.has_label() else None
        test_label = test.label_col if test.has_label() else None

        # If datasets have intersecting indexes:
        if set(train_data.index).intersection(set(test_data.index)):
            revert_index = True
            new_train_index = [f'train_{i}' for i in range(train_data.shape[0])]
            new_test_index = [f'test_{i}' for i in range(test_data.shape[0])]
            new_to_org_index_dict_train = dict(zip(new_train_index, list(train_data.index)))
            new_to_org_index_dict_test = dict(zip(new_test_index, list(test_data.index)))

            train_data.index = new_train_index
            test_data.index = new_test_index
            if train_label is not None:
                train_label.index = new_train_index
            if test_label is not None:
                test_label.index = new_test_index

        if drop_train is True:
            x, y = basic_preprocessing_for_domain_classification(train_data, test_data, cat_features,
                                                                 impute_numerical_nones=True)

            train_clusters = cluster_samples_by_label(x, y, self.random_state)
            drop_counter = 0

            redundant_samples = 0
            for node_i, samples_in_node in train_clusters.items():
                if samples_in_node:
                    train_samples_in_node = sorted(list(set(samples_in_node).intersection(set(train_data.index))))
                    node_ratio = len(train_samples_in_node) / len(samples_in_node)

                    if node_ratio > 0.5:
                        n_test_samples = len(samples_in_node) - len(train_samples_in_node)
                        n_train_samples = len(train_samples_in_node)
                        redundant_samples += n_train_samples - n_test_samples

            cutoff_factor = min(max_n_samples_to_drop / redundant_samples, 1)

            for node_i, samples_in_node in train_clusters.items():
                if drop_counter >= max_n_samples_to_drop:
                    break
                if samples_in_node:
                    train_samples_in_node = sorted(list(set(samples_in_node).intersection(set(train_data.index))))
                    node_ratio = len(train_samples_in_node) / len(samples_in_node)

                    if node_ratio > 0.5:
                        n_test_samples = len(samples_in_node) - len(train_samples_in_node)
                        n_train_samples = len(train_samples_in_node)
                        n_samples_to_drop = int((n_train_samples - n_test_samples) * cutoff_factor)
                        if n_samples_to_drop > max_n_samples_to_drop - drop_counter:
                            n_samples_to_drop = max_n_samples_to_drop - drop_counter
                        drop_counter += n_samples_to_drop

                        if n_samples_to_drop >= len(train_samples_in_node):
                            indices_to_drop = train_samples_in_node
                        else:
                            indices_to_drop = \
                                list(np.random.choice(train_samples_in_node, n_samples_to_drop, replace=False))
                        train_data.drop(indices_to_drop, inplace=True)
                        if train_label is not None:
                            train_label.drop(indices_to_drop, inplace=True)

        if oversample_train is True:
            # Cluster train and test data to find clusters with more test data than train data:
            x, y = basic_preprocessing_for_domain_classification(train_data, test_data, cat_features,
                                                                 impute_numerical_nones=True)

            train_clusters = cluster_samples_by_label(x, y, self.random_state)

            next_generated_i = 0

            cat_features_for_oversampling = copy(cat_features)
            if train_label is not None and context.task_type != TaskType.REGRESSION:
                cat_features_for_oversampling.append(train_label.name)

            # Over-sample data with the label:
            if train_label is not None:
                train_data = pd.concat([train_data, train_label], axis=1)
                if test_label is not None:
                    test_data = pd.concat([test_data, test_label], axis=1)
                else:
                    test_data = pd.concat(
                        [test_data, pd.Series([np.nan] * test_data.shape[0], index=test_data.index)],
                        axis=1)

            if use_smote is True:
                all_data = pd.concat([train_data, test_data])
                all_data[cat_features_for_oversampling] = all_data[cat_features_for_oversampling].astype(str)
                non_cat_cols = [col for col in all_data.columns if col not in cat_features_for_oversampling]
                all_data[non_cat_cols] = all_data[non_cat_cols].fillna(
                    {col: all_data[col].mean() for col in non_cat_cols})
                # cat features need to be given as indices:
                cat_indexes_for_smote = [all_data.columns.get_loc(cat_feature) for cat_feature in
                                         cat_features_for_oversampling]

            for node_i, samples_in_node in train_clusters.items():
                if samples_in_node:
                    train_samples_in_node = sorted(list(set(samples_in_node).intersection(set(train_data.index))))
                    node_ratio = len(train_samples_in_node) / len(samples_in_node)

                    if 0 < node_ratio < 0.5:
                        n_samples_to_add = int((0.5 - node_ratio) * len(samples_in_node))

                        if use_smote is False:
                            duplicate_indices_to_add = list(
                                np.random.choice(train_samples_in_node, n_samples_to_add, replace=True))
                            train_data = train_data.append(train_data.loc[duplicate_indices_to_add])
                        elif use_smote is True and n_samples_to_add > 0 and len(train_samples_in_node) > k_neighbors:
                            sm = SMOTENC(categorical_features=cat_indexes_for_smote,
                                         sampling_strategy={0: n_samples_to_add + len(train_samples_in_node)},
                                         k_neighbors=k_neighbors, random_state=self.random_state)
                            x_sm, y_sm = sm.fit_resample(all_data.loc[samples_in_node], y.loc[samples_in_node])
                            new_index = [f'generated_{i}' for i in
                                         range(next_generated_i, next_generated_i + n_samples_to_add)]
                            next_generated_i += n_samples_to_add
                            data_to_add = x_sm[y_sm == 0][-n_samples_to_add:]
                            data_to_add.index = new_index
                            train_data = train_data.append(data_to_add)

            # Separate the label:
            if train_label is not None:
                train_label = train_data[train_label.name]
                train_data = train_data.drop(train_label.name, axis=1)
                if test_label is not None:
                    test_label = test_data[test_label.name]
                    test_data = test_data.drop(test_label.name, axis=1)

            new_to_org_index_dict_train.update({f'generated_{i}': f'generated_{i}' for i in range(next_generated_i)})

        if move_from_test is True:
            train_preds, test_preds = calc_domain_preds(df_a=train_data, df_b=test_data,
                                                        cat_features=cat_features, random_state=self.random_state,
                                                        max_samples_for_training=self.n_samples)
            test_significant_samples = test_preds.sort_values(ascending=False)[
                test_preds > significant_threshold_test].sample(frac=0.5, random_state=self.random_state)
            n_samples_to_move = min(len(test_significant_samples), max_n_samples_to_move)
            test_samples_to_move = test_significant_samples.sample(n_samples_to_move,
                                                                   random_state=self.random_state).index

            train_data = pd.concat([train_data, test_data.loc[test_samples_to_move]])
            test_data = test_data.drop(test_samples_to_move)

            # If originally index had common values between train and test, add a unique prefix to test samples:
            if revert_index is True:
                new_to_org_index_dict_train.update({x: f'org_from_test_{x}' for x in test_samples_to_move})

            if train_label is not None:
                train_label = pd.concat([train_label, test_label.loc[test_samples_to_move]])
            if test_label is not None:
                test_label = test_label.drop(test_samples_to_move)

        # Rejoin label to data:
        if train_label is not None:
            train_data = pd.concat([train_data, train_label], axis=1)
        if test_label is not None:
            test_data = pd.concat([test_data, test_label], axis=1)

        if revert_index:
            train_data.index = [new_to_org_index_dict_train[x] for x in train_data.index]
            test_data.index = [new_to_org_index_dict_test[x] for x in test_data.index]

        return FixResult(fixed_train=train.copy(train_data), fixed_test=test.copy(test_data))


def calc_domain_preds(df_a, df_b, cat_features, random_state, max_samples_for_training):
    # sample size of train and test so that 2 domain classifiers can be trained on the data,
    # as we want to have viable predictions for all data samples (and can't use the domain classifier's predictions
    # on the data it was trained on)
    sample_size = min(max_samples_for_training, int(df_a.shape[0] / 2), int(df_b.shape[0] / 2))

    # create new dataset, with label denoting whether sample belongs to test dataset
    x, y = basic_preprocessing_for_domain_classification(df_a, df_b, cat_features)
    x_train_1, x_train_2, y_train_1, y_train_2 = train_test_split(x, y, train_size=2 * sample_size / x.shape[0],
                                                                  random_state=random_state, stratify=y)

    df_a_2_processed = x_train_2[(y_train_2 == 0).values]
    df_b_2_processed = x_train_2[(y_train_2 == 1).values]

    df_a_1_processed = x_train_1[(y_train_1 == 0).values]
    df_b_1_processed = x_train_1[(y_train_1 == 1).values]

    # train a model to disguise between train and test samples
    domain_classifier_1 = get_domain_classifier_hq(x_train=x_train_1, y_train=y_train_1,
                                                   cat_features=[col in cat_features for col in x.columns],
                                                   random_state=random_state)

    if x_train_2.shape[0] > 2 * sample_size:
        x_train_2, _, y_train_2, _ = train_test_split(x_train_2, y_train_2,
                                                      train_size=2 * sample_size / x_train_2.shape[0],
                                                      random_state=random_state, stratify=y_train_2)

    domain_classifier_2 = get_domain_classifier_hq(x_train=x_train_2, y_train=y_train_2,
                                                   cat_features=[col in cat_features for col in x.columns],
                                                   random_state=random_state)

    df_a_1_predictions = domain_classifier_1.predict_proba(df_a_1_processed)[:, 1]
    df_b_1_predictions = domain_classifier_1.predict_proba(df_b_1_processed)[:, 1]

    df_a_2_predictions = domain_classifier_2.predict_proba(df_a_2_processed)[:, 1]
    df_b_2_predictions = domain_classifier_2.predict_proba(df_b_2_processed)[:, 1]

    df_a_predictions = pd.Series(np.concatenate([df_a_1_predictions, df_a_2_predictions]),
                                 index=list(df_a_1_processed.index) + list(df_a_2_processed.index))
    df_b_predictions = pd.Series(np.concatenate([df_b_1_predictions, df_b_2_predictions]),
                                 index=list(df_b_1_processed.index) + list(df_b_2_processed.index))

    return df_a_predictions, df_b_predictions


def cluster_samples_by_label(x, y, random_state):
    model = DecisionTreeClassifier(max_depth=30, min_samples_leaf=50, random_state=random_state)
    model.fit(x, y)
    x_apply = model.apply(x)
    ret_dict = OrderedDict()
    for node_i in range(max(x_apply)):
        indexes_in_node = list(np.where(x_apply == node_i)[0])
        ret_dict[node_i] = list(x.iloc[indexes_in_node].index)
    return ret_dict