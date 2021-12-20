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
"""The UnusedFeatures check module."""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck, ConditionResult
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.validation import validate_model
from deepchecks.errors import DeepchecksValueError


__all__ = ['UnusedFeatures']


def naive_encoder(dataset: Dataset) -> TransformerMixin:
    """Create a naive encoder for categorical and numerical features.

    The encoder handles nans for all features and uses label encoder for categorical features. Then, all features are
    scaled using RobustScaler.

    Args:
        dataset: The dataset to encode.

    Returns:
        A transformer object.
    """
    numeric_features = list(set(dataset.features) - set(dataset.cat_features))

    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('nan_handling', SimpleImputer()),
                ('norm', RobustScaler())
            ]),
             numeric_features),
            ('cat',
             Pipeline([
                 ('nan_handling', SimpleImputer(strategy='most_frequent')),
                 ('encode', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)),
                 ('norm', RobustScaler())
             ]),
             dataset.cat_features)
        ]
    )


class UnusedFeatures(TrainTestBaseCheck):
    """Detect features that are nearly unused by the model.

    The check uses feature importance (either internally computed in appropriate models or calculated by permutation
    feature importance) to detect features that are not used by the model. From this list, the check sorts the features
    by their variance (as calculated by a PCA transformation). High variance unused features may be containing
     information that is ignored by the model.

    Args:
        feature_importance_threshold (float): A cutoff value for the feature importance, measured by the ratio of
            each features' feature importance to the mean feature importance. Features with lower importance
            are not shown in the check display.
        feature_variance_threshold (float): A cutoff value for the feature variance, measured by the ratio of
            each features' feature importance to the mean feature importance. Unused features with lower variance
            are not shown in the check display.
        n_top_fi_to_show (int): The max number of important features to show in the check display.
        n_top_unused_to_show (int): The max number of unused features to show in the check display, from among
            unused features that have higher variance then is defined by feature_variance_threshold.
        random_state (int): The random state to use for permutation feature importance and PCA.
    """

    def __init__(self, feature_importance_threshold: float = 0.2, feature_variance_threshold: float = 0.4,
                 n_top_fi_to_show: int = 5, n_top_unused_to_show: int = 15, random_state: int = 42):
        super().__init__()
        self.feature_importance_threshold = feature_importance_threshold
        self.feature_variance_threshold = feature_variance_threshold
        self.n_top_fi_to_show = n_top_fi_to_show
        self.n_top_unused_to_show = n_top_unused_to_show
        self.random_state = random_state

    def run(self, train_dataset: Dataset = None, test_dataset: Dataset = None, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label column. If test_dataset is not
                                     supplied this dataset will be used.
            test_dataset (Dataset): The test dataset object. Must contain a label column. Will be used if supplied.
            model: A scikit-learn-compatible fitted estimator instance

        Returns:
            CheckResult:
                value is a dataframe with metrics as indexes, and scores per training and test in the columns.
                display data is a bar graph of the metrics for training and test data.

        Raises:
            DeepchecksValueError: If neither train_dataset nor test_dataset exist, or either of the dataset objects are
                                  not a Dataset instance with a label.
        """
        if test_dataset:
            dataset = test_dataset
        elif train_dataset:
            dataset = train_dataset
        else:
            raise DeepchecksValueError('Either train_dataset or test_dataset must be supplied')
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        dataset.validate_label()
        validate_model(dataset, model)

        feature_importance = calculate_feature_importance(model, dataset,
                                                          permutation_wkargs={'random_state': self.random_state})

        # Calculate normalized variance per feature based on PCA decomposition
        pre_pca_transformer = naive_encoder(dataset)
        pca_trans = PCA(n_components=len(dataset.features) // 2, random_state=self.random_state)
        n_samples = min(10000, dataset.n_samples)
        pca_trans.fit(pre_pca_transformer.fit_transform(
            dataset.features_columns.sample(n_samples, random_state=self.random_state)
        ))

        feature_normed_variance = pd.Series(np.abs(pca_trans.components_).sum(axis=0), index=feature_importance.index)
        feature_normed_variance = feature_normed_variance / feature_normed_variance.sum()

        feature_df = pd.concat([feature_importance, feature_normed_variance], axis=1)
        feature_df.columns = ['Feature Importance', 'Feature Variance']
        feature_df.sort_values(by='Feature Importance', ascending=False, inplace=True)

        # For feature importance and variance, calculate their "ratio to average" per feature. The ratio to average
        # is, for example, the amount of feature importance a feature has, divided by the the amount he would have
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

            # limit display to n_top_to_show params
            display_feature_df = pd.concat(
                [feature_df.iloc[:(last_important_feature_index + 1)].head(self.n_top_fi_to_show),
                 unviable_feature_df.iloc[:last_variable_feature_index].head(self.n_top_unused_to_show)],
                axis=0)

            def plot_feature_importance():

                width = 0.20
                my_cmap = plt.cm.get_cmap('Set2')

                indices = np.arange(len(display_feature_df.index))

                colors = my_cmap(range(len(display_feature_df)))
                plt.figure(figsize=[8.0, 6.0 * len(display_feature_df) / 8.0])
                plt.barh(indices, display_feature_df['Feature Importance'].values.flatten(), height=width,
                         color=colors[0])
                plt.barh(indices + width, display_feature_df['Feature Variance'].values.flatten(), height=width,
                         color=colors[1])
                plt.xlabel('Importance / Variance [%]')
                plt.yticks(ticks=indices + width / 2., labels=display_feature_df.index)
                plt.yticks(rotation=30)
                last_important_feature_index_to_plot = min(last_important_feature_index, self.n_top_fi_to_show - 1)
                legend_labels = display_feature_df.columns.values.tolist()
                if last_important_feature_index_to_plot < len(display_feature_df) - 1:
                    last_important_feature_line_loc = last_important_feature_index_to_plot + 0.6
                    plt.plot(plt.gca().get_xlim(),
                             [last_important_feature_line_loc, last_important_feature_line_loc], 'k--')
                    legend_labels = ['Last shown significant feature'] + legend_labels
                plt.gca().invert_yaxis()
                plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.55, 1.02))
                plt.title('Unused features compared to top important features')

            # display only if high variance features exist (as set by self.feature_variance_threshold)
            if not last_variable_feature_index:
                display_list = []
            else:
                display_list = [
                    'Features above the line are a sample of the most important features, while the features '
                    'below the line are the unused features with highest variance, as defined by check'
                    ' parameters', plot_feature_importance]

        else:
            display_list = []

        return_value = {
            'used features': feature_df.index[:(last_important_feature_index + 1)].values.tolist(),
            'unused features': {
                'high variance': [] if unviable_feature_df.empty else unviable_feature_df.index[
                                                                      :last_variable_feature_index].values.tolist(),
                'low variance': [] if unviable_feature_df.empty else unviable_feature_df.index[
                                                                     last_variable_feature_index:].values.tolist()
            }}

        return CheckResult(return_value, header='Unused Features', display=display_list)

    def add_condition_number_of_high_variance_unused_features_not_greater_than(
            self, max_high_variance_unused_features: int = 5):
        """Add condition - require number of high variance unused features to be not greater than a given number.

        Args:
            max_high_variance_unused_features (int): Maximum allowed number of high variance unused features.
        """
        def max_high_variance_unused_features_condition(result: dict) -> ConditionResult:
            if len(result['unused features']['high variance']) > max_high_variance_unused_features:
                return ConditionResult(
                    False,
                    f'Found {result["unused features"]["high variance"]} unused high variance features')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Number of high variance unused features is not greater than'
                                  f' {max_high_variance_unused_features}',
                                  max_high_variance_unused_features_condition)
