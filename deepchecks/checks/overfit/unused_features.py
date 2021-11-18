"""The UnusedFeatures check module."""
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OrdinalEncoder

from deepchecks.feature_importance_utils import calculate_feature_importance
from deepchecks.utils import model_type_validation, DeepchecksValueError
from deepchecks import Dataset, CheckResult, TrainTestBaseCheck

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
    numeric_features = list(set(dataset.features()) - set(dataset.cat_features))

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
    """Detect features that are nearly unused by the model but have high variance.

    The check uses feature importance (either internally computed in appropriate models or calculated by permutation
    feature importance) to detect features that are not used by the model. From this list, the check displays the
    features that have high variance (as calculated by a PCA transformation). These features may be containing
    information that is ignored by the model.
    """

    def __init__(self, feature_importance_threshold: float = 0.2, feature_variance_threshold: float = 0.4):
        """Initialize the TrainTestDifferenceOverfit check.

        Args:
            feature_importance_threshold (float): An optional dictionary of metric name to scorer functions
            feature_variance_threshold (float): An optional dictionary of metric name to scorer functions
        """
        super().__init__()
        self.feature_importance_threshold = feature_importance_threshold
        self.feature_variance_threshold = feature_variance_threshold

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
                data is a bar graph of the metrics for training and test data.

        Raises:
            DeepchecksValueError: If neither train_dataset nor test_dataset exist, or either of the dataset objects are
                                  not a Dataset instance with a label.
        """
        func_name = self.__class__.__name__
        if test_dataset:
            dataset = test_dataset
        elif train_dataset:
            dataset = train_dataset
        else:
            raise DeepchecksValueError('Either train_dataset or test_dataset must be supplied')
        Dataset.validate_dataset(dataset, func_name)
        dataset.validate_label(func_name)
        test_dataset.validate_label(func_name)
        model_type_validation(model)

        feature_importance = calculate_feature_importance(model, dataset)

        # Calculate normalized variance per feature based on PCA decomposition
        pre_pca_transformer = naive_encoder(dataset)
        pca_trans = PCA(n_components=len(dataset.features()) // 2)
        n_samples = min(10000, dataset.n_samples())
        pca_trans.fit(pre_pca_transformer.fit_transform(dataset.features_columns().sample(n_samples)))

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
        if not unviable_feature_df.empty:
            unviable_feature_df.sort_values(by='Feature Variance', ascending=False, inplace=True)
            unviable_feature_ratio_to_avg_df = unviable_feature_df / (1 / len(feature_df))
            last_variable_feature_index = sum(
                unviable_feature_ratio_to_avg_df['Feature Variance'] > self.feature_variance_threshold
            )

            feature_df = pd.concat([feature_df.iloc[:(last_important_feature_index + 1)],
                                    unviable_feature_df.iloc[:(last_variable_feature_index + 1)]],
                                   axis=0)

        def plot_feature_importance():

            width = 0.20
            my_cmap = plt.cm.get_cmap('Set2')

            indices = np.arange(len(feature_df.index))

            colors = my_cmap(range(len(feature_df)))
            plt.figure(figsize=[8.0, 6.0 * len(feature_df) / 8.0])
            plt.barh(indices, feature_df['Feature Importance'].values.flatten(), height=width, color=colors[0])
            plt.barh(indices + width, feature_df['Feature Variance'].values.flatten(), height=width, color=colors[1])
            plt.xlabel('Importance / Variance [%]')
            plt.yticks(ticks=indices + width / 2., labels=feature_df.index)
            plt.yticks(rotation=30)
            legend_labels = feature_df.columns.values.tolist()
            if last_important_feature_index < len(feature_df) - 1:
                last_important_feature_line_loc = last_important_feature_index + 0.6
                plt.plot(plt.gca().get_xlim(),
                         [last_important_feature_line_loc, last_important_feature_line_loc], 'k--')
                legend_labels = ['Last significant feature'] + legend_labels
            plt.gca().invert_yaxis()
            plt.legend(legend_labels, loc='upper right', bbox_to_anchor=(1.55, 1.02))

        return CheckResult({}, check=self.__class__, header='Unused Features',
                           display=[plot_feature_importance])
