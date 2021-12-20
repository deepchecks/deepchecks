# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains the domain classifier drift check."""
from functools import partial
from typing import List
import warnings

import numpy as np
import pandas as pd

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.checks.distribution.dist_utils import preprocess_for_psi, drift_score_bar
from deepchecks.checks.distribution.plot import plot_density
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.utils.typing import Hashable

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from sklearn.experimental import enable_hist_gradient_boosting  # noqa # pylint: disable=unused-import
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

__all__ = ['WholeDatasetDrift']


class WholeDatasetDrift(TrainTestBaseCheck):
    """
    Calculate drift between the entire train and test datasets using a model trained to distinguish between them.

    Check fits a new model to distinguish between train and test datasets, called a Domain Classifier.
    Once the Domain Classifier is fitted the check calculates the feature importance for the domain classifier
    model. The result of the check is based on the AUC of the domain classifier model, and the check displays
    the change in distribution between train and test for the top features according to the
    calculated feature importance.

    Args:
        n_top_features (int):
            Amount of columns to show ordered by domain classifier feature importance. This limit is used together
            (AND) with min_feature_importance, so less than n_top_features features can be displayed.
        min_feature_importance (float): Minimum feature importance to show in the check display. Feature importance
            sums to 1, so for example the default value of 0.05 means that all features with importance contributing
            less than 5% to the predictive power of the Domain Classifier won't be displayed. This limit is used
            together (AND) with n_top_features, so features more important than min_feature_importance can be
            hidden.
        max_num_categories (int):
            Only for categorical columns. Max number of categories to display in distributio plots. If there are
            more, they are binned into an "Other" category in the display. If max_num_categories=None, there is
            no limit.
        sample_size (int):
            Max number of rows to use from each dataset for the training and evaluation of the domain classifier.
        random_state (int):
            Random seed for the check.
        test_size (float):
            Fraction of the combined datasets to use for the evaluation of the domain classifier.

    """

    def __init__(
            self,
            n_top_features: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories: int = 10,
            sample_size: int = 10000,
            random_state: int = 42,
            test_size: float = 0.3
    ):
        super().__init__()

        self._cat_features = None
        self.n_top_features = n_top_features
        self.min_feature_importance = min_feature_importance
        self.max_num_categories = max_num_categories
        self.sample_size = sample_size
        self.random_state = random_state
        self.test_size = test_size

    def run(self, train_dataset, test_dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object.
            test_dataset (Dataset): The test dataset object.
            model: not used in this check.

        Returns:
            CheckResult:
                value: dictionary containing the domain classifier auc and a dict of column name to its feature
                       importance as calculated for the domain classifier model.
                display: distribution graph for each column for the columns most explaining the dataset difference,
                         comparing the train and test distributions.

        Raises:
            DeepchecksValueError: If the object is not a Dataset or DataFrame instance
        """
        train_dataset = Dataset.validate_dataset_or_dataframe(train_dataset)
        test_dataset = Dataset.validate_dataset_or_dataframe(test_dataset)

        features = train_dataset.validate_shared_features(test_dataset)
        cat_features = train_dataset.validate_shared_categorical_features(test_dataset)
        self._cat_features = cat_features

        domain_classifier = self._generate_model(list(set(features) - set(cat_features)), cat_features)

        sample_size = min(self.sample_size, train_dataset.n_samples, test_dataset.n_samples)
        train_sample_df = train_dataset.features_columns.sample(sample_size, random_state=self.random_state)
        test_sample_df = test_dataset.features_columns.sample(sample_size, random_state=self.random_state)

        # create new dataset, with label denoting whether sample belongs to test dataset
        domain_class_df = pd.concat([train_sample_df, test_sample_df])
        domain_class_labels = pd.Series([0] * len(train_sample_df) + [1] * len(test_sample_df))

        x_train, x_test, y_train, y_test = train_test_split(domain_class_df, domain_class_labels,
                                                            stratify=domain_class_labels,
                                                            random_state=self.random_state,
                                                            test_size=self.test_size)

        domain_classifier = domain_classifier.fit(x_train, y_train)

        y_test.name = 'belongs_to_test'
        domain_test_dataset = Dataset(pd.concat([x_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1),
                                      cat_features=cat_features, label_name='belongs_to_test')

        # calculate feature importance of domain_classifier, containing the information which features separate
        # the dataset best.
        fi_ser = calculate_feature_importance(domain_classifier, domain_test_dataset, force_permutation=True,
                                              permutation_wkargs={'n_repeats': 10, 'random_state': self.random_state}
                                              ).sort_values(ascending=False)

        values_dict = {
            'domain_classifier_auc': roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1]),
            'domain_classifier_feature_importance': fi_ser.to_dict(),
        }

        headnote = """<span>
                    The shown features are the features that are most important for the domain classifier - the
                    domain_classifier trained to distinguish between the train and test datasets.<br> The percents of
                    explained dataset difference are the calculated feature importance values for the feature.
                </span><br><br>"""

        top_fi = fi_ser.head(self.n_top_features)
        top_fi = top_fi.loc[top_fi > self.min_feature_importance]

        def display_drift_score():
            plt.figure(figsize=(8, 0.5))
            drift_score_bar(plt.gca(), self.auc_to_drift_score(values_dict['domain_classifier_auc']),
                            'Whole dataset total')
            plt.figure(figsize=(8, 0.1))
            plt.axhline(y=0.5, color='k', linestyle='-', linewidth=0.5)
            plt.axis('off')

        displays = ([headnote] + [display_drift_score] + ['<h5>Main features contributing to drift</h5>'] +
                    [partial(self._display_dist, train_sample_df[feature], test_sample_df[feature], fi_ser)
                     for feature in top_fi.index]) if len(top_fi) else None

        return CheckResult(value=values_dict, display=displays, header='Whole Dataset Drift')

    @staticmethod
    def auc_to_drift_score(auc: float) -> float:
        """Calculate the drift score, which is 2*auc - 1, with auc being the auc of the Domain Classifier."""
        return max(2 * auc - 1, 0)

    def _display_dist(self, train_column: pd.Series, test_column: pd.Series, fi_ser: pd.Series):
        """Display a distribution comparison plot for the given columns."""
        colors = ['darkblue', '#69b3a2']

        plt.figure(figsize=(8, 3))
        axs = plt.gca()

        column_name = train_column.name

        if column_name in self._cat_features:
            expected_percents, actual_percents, categories_list = \
                preprocess_for_psi(dist1=train_column.dropna().values.reshape(-1),
                                   dist2=test_column.dropna().values.reshape(-1),
                                   max_num_categories=self.max_num_categories)

            cat_df = pd.DataFrame({'Train dataset': expected_percents, 'Test dataset': actual_percents},
                                  index=categories_list)

            cat_df.plot.bar(ax=axs, color=colors)
            axs.set_ylabel('Percentage')
            axs.legend()
            plt.xticks(rotation=30)

        else:
            x_range = (min(train_column.min(), test_column.min()), max(train_column.max(), test_column.max()))
            xs = np.linspace(x_range[0], x_range[1], 40)
            pdf1 = plot_density(train_column, xs, colors[0])
            pdf2 = plot_density(test_column, xs, colors[1])
            plt.gca().set_ylim(bottom=0, top=max(max(pdf1), max(pdf2)) * 1.1)
            axs.set_xlabel(column_name)
            axs.set_ylabel('Probability Density')
            axs.legend(['Train dataset', 'Test Dataset'])

        plt.title(f'Feature: {column_name} - Explains {format_percent(fi_ser.loc[column_name])} of dataset difference')

    def _generate_model(self, numerical_columns: List[Hashable], categorical_columns: List[Hashable]) -> Pipeline:
        """Generate the unfitted Domain Classifier model."""
        categorical_transformer = Pipeline(
            steps=[('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=np.nan,
                                              dtype=np.float64))]
        )

        preprocessor = ColumnTransformer(
            transformers=[
                ('num', 'passthrough', numerical_columns),
                ('cat', categorical_transformer, categorical_columns),
            ]
        )

        return Pipeline(
            steps=[('preprocessing', preprocessor),
                   ('model', HistGradientBoostingClassifier(
                       max_depth=2, max_iter=10, random_state=self.random_state,
                       categorical_features=[False] * len(numerical_columns) + [True] * len(categorical_columns)
                   ))])

    def add_condition_overall_drift_value_not_greater_than(self, max_drift_value: float = 0.25):
        """Add condition.

        Overall drift score, calculated as (2 * AUC - 1) for the AUC of the dataset discriminator model, is not greater
        than the specified value. This value is used as it scales the AUC value to the range [0, 1], where 0 indicates
        a random model (and no drift) and 1 indicates a perfect model (and completely distinguishable datasets).

        Args:
            max_drift_value (float): Maximal drift value allowed (value 0 and above)
        """

        def condition(result: dict):
            drift_score = self.auc_to_drift_score(result['domain_classifier_auc'])
            if drift_score > max_drift_value:
                message = f'Found drift value of: {format_number(drift_score)}, corresponding to a domain classifier ' \
                          f'AUC of: {format_number(result["domain_classifier_auc"])}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Drift value is not greater than {format_number(max_drift_value)}',
                                  condition)
