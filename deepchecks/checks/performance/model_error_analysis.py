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
"""Module of segment performance check."""
from functools import partial

import numpy as np
import pandas as pd
from category_encoders import TargetEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import plotly.express as px

from deepchecks import Dataset, CheckResult, TrainTestBaseCheck
from deepchecks.errors import DeepchecksProcessError
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.metrics import task_type_check, ModelType
from deepchecks.utils.plot import colors
from deepchecks.utils.strings import format_number
from deepchecks.utils.validation import validate_model

__all__ = ['ModelErrorAnalysis']


class ModelErrorAnalysis(TrainTestBaseCheck):
    """Find features that contribute to error in the model.

    Args:
        max_features (int): maximal number of features to show. (default: 3)
        min_feature_contribution (float): minimum contribution to the internal error model. (default: 0.15)
        min_error_model_score (float): minimum r^2 score for displaying the check. (default: 0.5)
        n_samples (int): number of samples to use for this check. (default: 50000)
        n_display_samples (int): number of samples to display. (default: 5000)
        random_seed (int): seed to calculate random from. (default: 42)
    """

    def __init__(
            self,
            max_features: int = 3,
            min_feature_contribution: float = 0.15,
            min_error_model_score: float = 0.5,
            n_samples: int = 50_000,
            n_display_samples: int = 5_000,
            random_seed: int = 42
    ):
        super().__init__()
        self.max_features = max_features
        self.min_feature_contribution = min_feature_contribution
        self.min_error_model_score = min_error_model_score
        self.n_samples = n_samples
        self.n_display_samples = n_display_samples
        self.random_state = random_seed

    def run(self, train_dataset: Dataset, test_dataset: Dataset, model=None) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        """
        # Validations
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        train_dataset.validate_shared_label(test_dataset)
        train_dataset.validate_shared_features(test_dataset)
        train_dataset.validate_shared_categorical_features(test_dataset)
        validate_model(train_dataset, model)

        task_type = task_type_check(model, train_dataset)

        cat_features = train_dataset.cat_features

        train_dataset = train_dataset.sample(self.n_samples, random_state=self.random_state)
        test_dataset = test_dataset.sample(self.n_samples, random_state=self.random_state)

        # Create scoring function, used to calculate the per sample model error
        if task_type == ModelType.REGRESSION:
            def scoring_func(dataset):
                return per_sample_mse(dataset.label_col, model.predict(dataset.features_columns))
        else:
            def scoring_func(dataset):
                return per_sample_binary_cross_entropy(dataset.label_col,
                                                       model.predict_proba(dataset.features_columns))

        train_scores = scoring_func(train_dataset)
        test_scores = scoring_func(test_dataset)

        # Create and fit model to predict the per sample error
        error_model = create_error_model(train_dataset, random_state=self.random_state)
        error_model.fit(train_dataset.features_columns, y=train_scores)

        # Check if fitted model is good enough
        error_model_predicted = error_model.predict(test_dataset.features_columns)
        error_model_score = r2_score(test_scores, error_model_predicted)

        # This check should be ignored if no information gained from the error model (low r2_score)
        if error_model_score < self.min_error_model_score:
            raise DeepchecksProcessError(f'Unable to train meaningful error model '
                                         f'(r^2 score: {format_number(error_model_score)})')

        error_fi = calculate_feature_importance(error_model, test_dataset)
        error_fi.sort_values(ascending=False, inplace=True)

        n_samples_display = min(self.n_display_samples, len(test_dataset))
        display_data = test_dataset.data
        display_error = pd.Series(error_model_predicted, name='model error', index=test_dataset.data.index)

        display = []

        for feature in error_fi.keys()[:self.max_features]:
            if error_fi[feature] < self.min_feature_contribution:
                break

            data = pd.concat([display_data[feature], display_error], axis=1)

            if feature in cat_features:
                display.append(px.violin(data, y='model error', x=feature, title=feature, box=False,
                                         color_discrete_sequence=[colors['Test']] * data[feature].nunique()))
            else:
                data = data.sample(n_samples_display, random_state=self.random_state)
                display.append(px.scatter(data, x=feature, y='model error', color='model error', title=feature))
        value = None
        headnote = """<span>
            The following graphs show the top features that are most useful for distinguishing high error 
            samples from low error samples. 
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)


def per_sample_binary_cross_entropy(y_true, y_pred):
    y_true = np.array(y_true)
    return - (np.tile(y_true.reshape((-1, 1)), (1, y_pred.shape[1])) *
              np.log(y_pred + np.finfo(float).eps)).sum(axis=1)


def per_sample_mse(y_true, y_pred):
    return (y_true - y_pred)**2


def create_error_model(dataset: Dataset, random_state=42):
    cat_features = dataset.cat_features
    numeric_features = [num_feature for num_feature in dataset.features if num_feature not in cat_features]

    numeric_transformer = SimpleImputer()
    categorical_transformer = Pipeline(
        steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', TargetEncoder(cat_features))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, cat_features),
        ]
    )

    return Pipeline(steps=[
        ('preprocessing', preprocessor),
        ('model', RandomForestRegressor(max_depth=4, n_jobs=-1, random_state=random_state))
    ])
