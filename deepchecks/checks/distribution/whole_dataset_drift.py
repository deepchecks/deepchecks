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
from typing import List
import warnings

import numpy as np
import pandas as pd

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.distribution.plot import feature_distribution_traces, drift_score_bar_traces
from deepchecks.utils.features import N_TOP_MESSAGE, calculate_feature_importance_or_none
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
import plotly.graph_objects as go


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
        n_top_columns (int):
            Amount of columns to show ordered by domain classifier feature importance. This limit is used together
            (AND) with min_feature_importance, so less than n_top_columns features can be displayed.
        min_feature_importance (float): Minimum feature importance to show in the check display. Feature importance
            sums to 1, so for example the default value of 0.05 means that all features with importance contributing
            less than 5% to the predictive power of the Domain Classifier won't be displayed. This limit is used
            together (AND) with n_top_columns, so features more important than min_feature_importance can be
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
            n_top_columns: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories: int = 10,
            sample_size: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3
    ):
        super().__init__()

        self._cat_features = None
        self.n_top_columns = n_top_columns
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
        train_dataset = Dataset.ensure_not_empty_dataset(train_dataset, cast=True)
        test_dataset = Dataset.ensure_not_empty_dataset(test_dataset, cast=True)

        features = self._datasets_share_features([train_dataset, test_dataset])
        cat_features = self._datasets_share_categorical_features([train_dataset, test_dataset])
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
                                      cat_features=cat_features, label='belongs_to_test')

        # calculate feature importance of domain_classifier, containing the information which features separate
        # the dataset best.
        fi, importance_type = calculate_feature_importance_or_none(
            domain_classifier,
            domain_test_dataset,
            force_permutation=True,
            permutation_kwargs={'n_repeats': 10, 'random_state': self.random_state},
            return_calculation_type=True
        )

        fi = fi.sort_values(ascending=False) if fi is not None else None

        domain_classifier_auc = roc_auc_score(y_test, domain_classifier.predict_proba(x_test)[:, 1])

        values_dict = {
            'domain_classifier_auc': domain_classifier_auc,
            'domain_classifier_drift_score': self.auc_to_drift_score(domain_classifier_auc),
            'domain_classifier_feature_importance': fi.to_dict() if fi is not None else {},
        }

        headnote = f"""
        <span>
        The shown features are the features that are most important for the domain classifier - the
        domain_classifier trained to distinguish between the train and test datasets.<br> The percents of
        explained dataset difference are the importance values for the feature calculated using `{importance_type}`.
        </span><br><br>
        """

        if fi is not None:
            top_fi = fi.head(self.n_top_columns)
            top_fi = top_fi.loc[top_fi > self.min_feature_importance]
        else:
            top_fi = None

        if top_fi is not None and len(top_fi):
            score = values_dict['domain_classifier_drift_score']

            displays = [headnote, self._build_drift_plot(score),
                        '<h3>Main features contributing to drift</h3>',
                        N_TOP_MESSAGE % self.n_top_columns]
            displays += [self._display_dist(train_sample_df[feature], test_sample_df[feature], top_fi)
                         for feature in top_fi.index]
        else:
            displays = None

        return CheckResult(value=values_dict, display=displays, header='Whole Dataset Drift')

    @staticmethod
    def auc_to_drift_score(auc: float) -> float:
        """Calculate the drift score, which is 2*auc - 1, with auc being the auc of the Domain Classifier."""
        return max(2 * auc - 1, 0)

    def _build_drift_plot(self, score):
        """Build traffic light drift plot."""
        bar_traces, x_axis, y_axis = drift_score_bar_traces(score)
        x_axis['title'] = 'Drift score'
        drift_plot = go.Figure(layout=dict(
            title='Drift Score - Whole Dataset Total',
            xaxis=x_axis,
            yaxis=y_axis,
            width=700,
            height=200

        ))

        drift_plot.add_traces(bar_traces)
        return drift_plot

    def _display_dist(self, train_column: pd.Series, test_column: pd.Series, fi_ser: pd.Series):
        """Display a distribution comparison plot for the given columns."""
        column_name = train_column.name

        title = f'Feature: {column_name} - Explains {format_percent(fi_ser.loc[column_name])} of dataset difference'
        traces, xaxis_layout, yaxis_layout = \
            feature_distribution_traces(train_column.dropna(),
                                        test_column.dropna(),
                                        is_categorical=column_name in self._cat_features,
                                        max_num_categories=self.max_num_categories)

        figure = go.Figure(layout=go.Layout(
            title=title,
            xaxis=xaxis_layout,
            yaxis=yaxis_layout,
            legend=dict(
                title='Dataset',
                yanchor='top',
                y=0.9,
                xanchor='left'),
            width=700,
            height=300
        ))

        figure.add_traces(traces)

        return figure

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
            drift_score = result['domain_classifier_drift_score']
            if drift_score > max_drift_value:
                message = f'Found drift value of: {format_number(drift_score)}, corresponding to a domain classifier ' \
                          f'AUC of: {format_number(result["domain_classifier_auc"])}'
                return ConditionResult(False, message)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Drift value is not greater than {format_number(max_drift_value)}',
                                  condition)
