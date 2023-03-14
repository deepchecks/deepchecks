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
import pandas as pd

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult, DatasetKind
from deepchecks.core.check_utils.multivariate_drift_utils import (get_domain_classifier_hq,
                                                                  preprocess_for_domain_classifier,
                                                                  run_multivariable_drift)
from deepchecks.core.fix_classes import TrainTestCheckFixMixin
from deepchecks.tabular import Context, TrainTestCheck
from deepchecks.utils.strings import format_number

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

        sample_size = min(self.n_samples, train_dataset.n_samples, test_dataset.n_samples)

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

    def fix_logic(self, context: Context, check_result: CheckResult, fix_method='move_to_train') -> Context:
        """Run fix.

        Parameters
        ----------
        context : Context
            Context object.
        check_result : CheckResult
            CheckResult object.
        fix_method : str, default: 'move_to_train'
            Method to fix the problem. Possible values: 'drop_features', 'replace_with_none', 'move_to_train'.
        """
        significant_threshold_test = 0.75
        significant_threshold_train = 0.4

        train, test = context.train, context.test
        cat_features, numerical_features = train.cat_features, train.numerical_features

        # sample size of train and test so that 2 domain classifiers can be trained on the data,
        # as we want to have viable predictions for all data samples (and can't use the domain classifier's predictions
        # on the data it was trained on)
        sample_size = min(self.n_samples, int(train.n_samples / 2), int(test.n_samples / 2))

        train_data = train.data[numerical_features + cat_features]
        test_data = test.data[numerical_features + cat_features]

        # create new dataset, with label denoting whether sample belongs to test dataset
        x, y = preprocess_for_domain_classifier(train_data, test_data, cat_features)

        train_processed = x[(y == 0).values]
        test_processed = x[(y == 1).values]

        # train a model to disguise between train and test samples
        domain_classifier = get_domain_classifier_hq(x_train=x, y_train=y,
                                                     cat_features=[col in cat_features for col in x.columns],
                                                     random_state=self.random_state)

        print(f'Before starting: {domain_classifier.score(x, y)}')

        test_predictions = domain_classifier.predict_proba(test_processed)[:, 1]
        test_samples_to_move = test_processed[test_predictions > significant_threshold_test] \
            .sample(frac=0.5, random_state=self.random_state).index
        train_data = pd.concat([train_data, test_data.loc[test_samples_to_move]])
        test_data = test_data.drop(test_samples_to_move)

        # Again train a model to differentiate between train and test samples, now in order to decide which train
        # samples to drop:
        # create new dataset, with label denoting whether sample belongs to test dataset
        train_data_to_train = train_data.sample(sample_size, random_state=self.random_state)
        test_data_to_train = test_data.sample(sample_size, random_state=self.random_state)

        x, y = preprocess_for_domain_classifier(train_data_to_train, test_data_to_train, cat_features)

        # train a model to disguise between train and test samples
        domain_classifier = get_domain_classifier_hq(x_train=x, y_train=y,
                                                     cat_features=[col in cat_features for col in x.columns],
                                                     random_state=self.random_state)

        print(f'After cleaning test samples: {domain_classifier.score(x, y)}')

        train_processed = x[(y == 0).values]
        test_processed = x[(y == 1).values]

        train_predictions = domain_classifier.predict_proba(train_processed)[:, 1]
        train_samples_to_drop = train_processed[train_predictions < significant_threshold_train].index
        train_data = train_data.drop(train_samples_to_drop)

        train_data_to_train = train_data.sample(sample_size, random_state=self.random_state)
        test_data_to_train = test_data.sample(sample_size, random_state=self.random_state)

        x, y = preprocess_for_domain_classifier(train_data_to_train, test_data_to_train, cat_features)

        # train a model to disguise between train and test samples
        domain_classifier = get_domain_classifier_hq(x_train=x, y_train=y,
                                                     cat_features=[col in cat_features for col in x.columns],
                                                     random_state=self.random_state)

        print(f'After cleaning train samples: {domain_classifier.score(x, y)}')

        # create new datasets
        context.set_dataset_by_kind(DatasetKind.TRAIN, context.train.copy(train_data))
        context.set_dataset_by_kind(DatasetKind.TEST, context.test.copy(test_data))

        return context

    @property
    def fix_params(self):
        """Return fix params for display."""
        return {'fix_method': {'display': 'Fix By',
                               'params': ['drop_features', 'replace_with_nones', 'move_to_train'],
                               'params_display': ['Dropping Features', 'Replacing With Nones',
                                                  'Moving Samples To Train'],
                               'params_description': ['Drop features with new categories from the dataset',
                                                      'Replace new categories with None values',
                                                      'Move samples with new categories from test to train']},
                'max_ratio': {'display': 'Max Ratio Of New Categories',
                              'params': float,
                              'params_description': 'Maximum ratio of samples with new categories'},
                'percentage_to_move': {'display': 'Percentage Of Samples To Move',
                                       'params': float,
                                       'params_description': 'Percentage of samples with new categories to move to '
                                                             'train. Relevant only for move_to_train fix method.'}}

    @property
    def problem_description(self):
        """Return problem description."""
        return """New categories are present in test data but not in train data. This can lead to wrong predictions in 
                  test data, as these new categories were unknown to the model during training."""

    @property
    def manual_solution_description(self):
        """Return manual solution description."""
        return """Resample train data to include samples with these new categories."""

    @property
    def automatic_solution_description(self):
        """Return automatic solution description."""
        return """There are 4 possible automatic solutions:
                  1. Drop features with new categories from the dataset, so the model won't train on them
                  3. Replace new categories with None values, so the model will treat them as missing values.
                     This is not recommended, as it doesn't solve the underlying issue.
                  4. Move samples with new categories from test to train.
                     This is the recommended solution, as it allows the model to be trained on the correct data.
                     However, this solution can cause data leakage, and should only be used if it's possible for train
                     dataset to have samples from test.
                     For example, if train and test datasets are from 2 different time periods, it would probably be
                     considered leakage if test samples (from the later time period) were moved to train."""
