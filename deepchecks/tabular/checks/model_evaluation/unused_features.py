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
"""The UnusedFeatures check module."""
from typing import Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, RobustScaler

from deepchecks.core import CheckResult, ConditionCategory, ConditionResult
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Context, Dataset, TrainTestCheck
from deepchecks.utils.function import run_available_kwargs

__all__ = ['UnusedFeatures']


class UnusedFeatures(TrainTestCheck):
    """Detect features that are nearly unused by the model.

    The check uses feature importance (either internally computed in appropriate models or calculated by permutation
    feature importance) to detect features that are not used by the model. From this list, the check sorts the features
    by their variance (as calculated by a PCA transformation). High variance unused features may be containing
    information that is ignored by the model.

    Parameters
    ----------
    feature_importance_threshold : float , default: 0.2
        A cutoff value for the feature importance, measured by the ratio of
        each features' feature importance to the mean feature importance. Features with lower importance
        are not shown in the check display.
    feature_variance_threshold : float , default: 0.4
        A cutoff value for the feature variance, measured by the ratio of
        each features' feature variance to the mean feature variance. Unused features with lower variance
        are not shown in the check display.
    n_top_fi_to_show : int , default: 5
        The max number of important features to show in the check display.
    n_top_unused_to_show : int , default: 15
        The max number of unused features to show in the check display, from among
        unused features that have higher variance then is defined by feature_variance_threshold.
    random_state : int , default: 42
        The random state to use for permutation feature importance and PCA.
    """

    def __init__(
        self,
        feature_importance_threshold: float = 0.2,
        feature_variance_threshold: float = 0.4,
        n_top_fi_to_show: int = 5,
        n_top_unused_to_show: int = 15,
        random_state: int = 42,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.feature_importance_threshold = feature_importance_threshold
        self.feature_variance_threshold = feature_variance_threshold
        self.n_top_fi_to_show = n_top_fi_to_show
        self.n_top_unused_to_show = n_top_unused_to_show
        self.random_state = random_state

    def run_logic(self, context: Context) -> CheckResult:
        """Run check."""
        if context.have_test():
            dataset = context.test
        else:
            dataset = context.train
        _ = context.model  # validate model

        feature_importance = context.feature_importance
        if feature_importance is None:
            raise DeepchecksValueError('Feature Importance is not available.')
        dataset.assert_features()

        # Calculate normalized variance per feature based on PCA decomposition
        pre_pca_transformer, var_col_order = naive_encoder(dataset)
        pca_trans = PCA(n_components=len(var_col_order) // 2, random_state=self.random_state)

        n_samples = min(10000, dataset.n_samples)
        fit_data = dataset.features_columns[var_col_order].sample(n_samples, random_state=self.random_state)
        pca_trans.fit(pre_pca_transformer.fit_transform(fit_data.fillna(0)))

        feature_normed_variance = pd.Series(np.abs(pca_trans.components_).sum(axis=0), index=var_col_order)
        feature_normed_variance = feature_normed_variance / feature_normed_variance.sum()

        feature_df = pd.concat([feature_importance, feature_normed_variance], axis=1)
        feature_df.columns = ['Feature Importance', 'Feature Variance']
        feature_df.sort_values(by='Feature Importance', ascending=False, inplace=True)

        # For feature importance and variance, calculate their "ratio to average" per feature. The ratio to average
        # is, for example, the amount of feature importance a feature has, divided by the amount he would have
        # if all features where equally important (which is basically 1 / n_of_features).
        feature_ratio_to_avg_df = feature_df / (1 / len(feature_importance))

        # Find last viable feature (not unused). All features from there on are sorted by variance
        last_important_feature_index = sum(
            feature_ratio_to_avg_df['Feature Importance'] > self.feature_importance_threshold
        ) - 1

        unviable_feature_df = feature_df.iloc[(last_important_feature_index + 1):]
        # Only display if there are features considered unimportant
        if not unviable_feature_df.empty:
            unviable_feature_df.sort_values(by='Feature Variance', ascending=False, inplace=True)
            unviable_feature_ratio_to_avg_df = unviable_feature_df / (1 / len(feature_df))
            last_variable_feature_index = sum(
                unviable_feature_ratio_to_avg_df['Feature Variance'] > self.feature_variance_threshold
            )

            # display only if high variance features exist (as set by self.feature_variance_threshold)
            if context.with_display and last_variable_feature_index:
                # limit display to n_top_to_show params
                display_feature_df = pd.concat(
                    [feature_df.iloc[:(last_important_feature_index + 1)].head(self.n_top_fi_to_show),
                     unviable_feature_df.iloc[:last_variable_feature_index].head(self.n_top_unused_to_show)],
                    axis=0)

                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=display_feature_df.index,
                    x=display_feature_df['Feature Importance'].multiply(100).values.flatten(),
                    name='Feature Importance %',
                    marker_color='indianred',
                    orientation='h'
                ))
                fig.add_trace(go.Bar(
                    y=display_feature_df.index,
                    x=display_feature_df['Feature Variance'].multiply(100).values.flatten(),
                    name='Feature Variance %',
                    marker_color='lightsalmon',
                    orientation='h'
                ))

                fig.update_yaxes(autorange='reversed')
                fig.update_layout(
                    title_text='Unused features compared to top important features',
                    height=500
                )

                last_important_feature_index_to_plot = min(last_important_feature_index, self.n_top_fi_to_show - 1)

                if last_important_feature_index_to_plot < len(display_feature_df) - 1:
                    last_important_feature_line_loc = last_important_feature_index_to_plot + 0.5
                    fig.add_hline(y=last_important_feature_line_loc, line_width=2, line_dash='dash', line_color='green',
                                  annotation_text='Last shown significant feature')
                display_list = [
                    'Features above the line are a sample of the most important features, while the features '
                    'below the line are the unused features with highest variance, as defined by check'
                    ' parameters', fig]
            else:
                display_list = []

        else:
            display_list = []

        return_value = {
            'used features': feature_df.index[:(last_important_feature_index + 1)].values.tolist(),
            'unused features': {
                'high variance': (
                    [] if unviable_feature_df.empty
                    else unviable_feature_df.index[:last_variable_feature_index].values.tolist()
                ),
                'low variance': (
                    [] if unviable_feature_df.empty
                    else unviable_feature_df.index[last_variable_feature_index:].values.tolist()
                )
            }}

        return CheckResult(return_value, header='Unused Features', display=display_list)

    def add_condition_number_of_high_variance_unused_features_less_or_equal(
            self, max_high_variance_unused_features: int = 5):
        """Add condition - require number of high variance unused features to be less or equal to threshold.

        Parameters
        ----------
        max_high_variance_unused_features : int , default: 5
            Maximum allowed number of high variance unused features.
        """
        def max_high_variance_unused_features_condition(result: dict) -> ConditionResult:
            high_var_features = result['unused features']['high variance']
            details = f'Found {len(high_var_features)} high variance unused features'
            category = ConditionCategory.PASS if len(high_var_features) <= max_high_variance_unused_features else \
                ConditionCategory.WARN
            return ConditionResult(category, details)

        return self.add_condition(f'Number of high variance unused features is less or equal to'
                                  f' {max_high_variance_unused_features}',
                                  max_high_variance_unused_features_condition)


def naive_encoder(dataset: Dataset) -> Tuple[TransformerMixin, list]:
    """Create a naive encoder for categorical and numerical features.

    The encoder handles nans for all features and uses label encoder for categorical features. Then, all features are
    scaled using RobustScaler.

    Parameters
    ----------
    dataset : Dataset
        The dataset to encode.

    Returns
    -------
    Tuple[TransformerMixin, list]
        A transformer object, a list of columns returned
    """
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('nan_handling', SimpleImputer()),
                ('norm', RobustScaler())
            ]),
                np.array(dataset.numerical_features, dtype='object')),
            ('cat',
             Pipeline([
                 ('nan_handling', SimpleImputer(strategy='most_frequent')),
                 ('encode', run_available_kwargs(OrdinalEncoder, handle_unknown='use_encoded_value', unknown_value=-1)),
                 ('norm', RobustScaler())
             ]),
             np.array(dataset.cat_features, dtype='object'))
        ]
    ), dataset.numerical_features + dataset.cat_features
