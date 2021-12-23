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
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.base import is_classifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import log_loss, mean_squared_error, r2_score
from category_encoders import TargetEncoder

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.errors import DeepchecksProcessError
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.strings import format_number
from deepchecks.utils.validation import validate_model


__all__ = ['ModelErrorAnalysis']


class ModelErrorAnalysis(SingleDatasetBaseCheck):
    """Find features that contribute to error in the model.

    Args:
        max_features (int): maximal number of features to show. (default: 3)
        min_feature_contribution (float): minimum contribution to the internal error model. (default: 0.15)
        min_error_model_score (float): minimum r^2 score for displaying the check. (default: 0.5)
        n_samples (int): number of samples to use for this check. (default: 50000)
        random_seed (int): seed to calculate random from. (default: 42)
    """

    def __init__(
        self,
        max_features: int = 3,
        min_feature_contribution: float = 0.15,
        min_error_model_score: float = 0.5,
        n_samples: int = 50_000,
        random_seed: int = 42
    ):
        super().__init__()
        self.max_features = max_features
        self.min_error = min_feature_contribution
        self.min_error_model_score = min_error_model_score
        self.n_samples = n_samples
        self.random_state = random_seed

    def run(self, dataset: Dataset, model) -> CheckResult:
        """Run check.

        Args:
            dataset (Dataset): a Dataset object.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        """
        # Validations
        Dataset.validate_dataset(dataset)
        dataset.validate_label()
        validate_model(dataset, model)

        cat_features = dataset.cat_features

        dataset = dataset.sample(self.n_samples, random_state=self.random_state)

        if is_classifier(model):
            y_pred = model.predict_proba(dataset.features_columns)
            labels = sorted(np.unique(dataset.label_col))
            score = list(map(lambda x, y: log_loss([x], [y], labels=labels), dataset.label_col, y_pred))
        else:
            y_pred = model.predict(dataset.features_columns)
            score = list(map(lambda x, y: mean_squared_error([x], [y]), dataset.label_col, y_pred))

        error_model = create_error_model(dataset, random_state=self.random_state)

        error_model_train_x, error_model_test_x, error_model_train_y, error_model_test_y = \
            train_test_split(dataset.features_columns, score, random_state=self.random_state)

        error_model.fit(error_model_train_x, y=error_model_train_y)

        error_model_predicted = error_model.predict(error_model_test_x)

        error_model_score = r2_score(error_model_predicted, error_model_test_y)

        # This check should be ignored if no information gained from the error model (low r2_score)
        if error_model_score < self.min_error_model_score:
            raise DeepchecksProcessError(f'Unable to train meaningful error model '
                                         f'(r^2 score: {format_number(error_model_score)})')

        error_fi = calculate_feature_importance(error_model, dataset)
        error_fi.sort_values(ascending=False, inplace=True)

        display_data = dataset.data.assign(score=score)

        min_score = min(score)
        max_score = max(score)

        def display_categorical(data, feature_name, groupby):
            plt.figure(figsize=(10, 7))
            ax = plt.gca()
            categories = []
            all_values = []
            for category, rows in groupby.groups.items():
                cat_data = data.iloc[rows]
                all_values.append(cat_data['score'])
                categories.append(category)

            ax.violinplot(all_values, showextrema=False,showmedians=True)
            ax.xaxis.set_major_locator(mticker.FixedLocator(range(0, len(categories) + 1)))
            ax.set_xticklabels([''] + categories)

            ax.set_ylabel('error score')
            ax.set_xlabel(feature_name)

            plt.xticks(rotation=30)
            plt.title(feature_name)

        def display_numeric(data, feature_name):
            plt.figure(figsize=(10, 7))
            ax = plt.gca()
            cm = plt.cm.get_cmap('RdYlBu_r')

            sc = plt.scatter(x=feature_name, y='score', data=data, alpha=0.1, edgecolors='none', s=10,
                             c=data['score'], vmin=min_score, vmax=max_score, cmap=cm)
            ax.set_ylabel('error score')
            ax.set_xlabel(feature_name)
            color_bar = plt.colorbar(sc)
            color_bar.set_alpha(1)
            color_bar.draw_all()

            plt.title(feature_name)

        display = []

        for feature in error_fi.keys()[:self.max_features]:
            if error_fi[feature] < self.min_error:
                break

            if feature in cat_features:
                feat = dataset.data[feature]
                scored_feature = pd.DataFrame(feat).assign(score=score)
                grouped_category = scored_feature.groupby(feat)
                display.append(partial(display_categorical, display_data, feature, grouped_category))
            else:
                display.append(partial(display_numeric, display_data, feature))

        value = None
        headnote = """<span>
            The following graphs show the top features that are most useful for distinguishing high error 
            samples from low error samples. 
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)


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
