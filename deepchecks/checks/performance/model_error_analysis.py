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
from typing import Callable, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.base import is_classifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.metrics import log_loss, mean_squared_error

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.validation import validate_model
from deepchecks.utils.typing import Hashable


__all__ = ['ModelErrorAnalysis']


class ModelErrorAnalysis(SingleDatasetBaseCheck):
    """Top features that contribute to error in the model.

    Args:
        max_segments (int): maximal number of segments to split the a values into.
    """

    feature_1: Optional[Hashable]
    feature_2: Optional[Hashable]
    metric: Union[str, Callable, None]
    max_segments: int

    def __init__(
        self,
        max_features: int = 3,
        min_error: float = 0.15
    ):
        super().__init__()
        self.max_features = max_features
        self.min_error = min_error

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

        if is_classifier(model):
            y_pred = model.predict_proba(dataset.features_columns)
            labels = sorted(np.unique(dataset.label_col))
            score = list(map(lambda x, y: log_loss([x], [y], labels=labels), dataset.label_col, y_pred))
        else:
            y_pred = model.predict(dataset.features_columns)
            score = pd.DataFrame({'label': dataset.label_col, 'pred': y_pred}).apply(
                lambda x: mean_squared_error(x['label'], y['pred']))

        numeric_transformer = SimpleImputer()
        categorical_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", TargetEncoder(cat_features))]
        )

        numeric_features = [num_feature for num_feature in dataset.features if num_feature not in dataset.cat_features]

        preprocessor = ColumnTransformer(
            transformers=[
                ("num", numeric_transformer, numeric_features),
                ("cat", categorical_transformer, cat_features),
            ]
        )

        error_model = Pipeline(steps=[
            ("preprocessing", preprocessor),
            ("model", RandomForestRegressor(max_depth=4, n_jobs=-1))
        ])

        error_model.fit(dataset.features_columns, y=score)

        error_fi = calculate_feature_importance(error_model, dataset)
        error_fi.sort_values(ascending=False, inplace=True)

        display_data = dataset.data.assign(score=score)

        def display_categorical(data, feature, groupby):
            fig = plt.figure(figsize=(10, 7))
            ax = plt.gca()
            categories = []
            all_values = []
            for category, rows in groupby.groups.items():
                cat_data = data.iloc[rows]
                all_values.append(cat_data['score'])
                categories.append(category)

            bp = ax.violinplot(all_values, showmedians=True)

            ax.set_xticklabels([0]+ categories)
            plt.xticks(rotation=30)
            plt.title(feature)

        def display_numeric(data, feature):
            fig = plt.figure(figsize=(10, 7))
            plt.scatter(x=feature, y='score', data=data)
            plt.title(feature)

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
            The following graphs show the top features that contribute to error and their values compared to the error
        </span>"""
        display = [headnote] + display if display else None

        return CheckResult(value, display=display)
