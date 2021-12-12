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
from typing import Callable, Union, Optional

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.axes import Axes
import seaborn as sns
from sklearn.base import is_classifier

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor
from category_encoders import TargetEncoder
from sklearn.metrics import log_loss, mean_squared_error

from deepchecks import Dataset, CheckResult, SingleDatasetBaseCheck
from deepchecks.checks.performance.partition import partition_column
from deepchecks.utils.metrics import validate_scorer, task_type_check, DEFAULT_SINGLE_METRIC, DEFAULT_METRICS_DICT
from deepchecks.utils.strings import format_number
from deepchecks.utils.features import calculate_feature_importance
from deepchecks.utils.validation import validate_model
from deepchecks.utils.typing import Hashable
from deepchecks.errors import DeepchecksValueError


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
        Dataset.validate_dataset(dataset, self.__class__.__name__)
        dataset.validate_label(self.__class__.__name__)
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

        display_params = {'data': dataset.data.assign(score=score), 'features': []}

        for feature in error_fi.keys()[:self.max_features]:
            if error_fi[feature] < self.min_error:
                break

            if feature in cat_features:
                feat = dataset.data[feature]
                scored_feature = pd.DataFrame(feat).assign(score=score)
                grouped_category = scored_feature.groupby(feat)
                params = {
                    'feature': feature,
                    'groupby': grouped_category,
                    'type': 'category'
                }
                display_params['features'].append(params)
            else:
                params = {
                    'feature': feature,
                    'type': 'numeric'
                }
                display_params['features'].append(params)

        def display():
            data = display_params['data']

            for display_feature in display_params['features']:
                if display_feature['type'] == 'category':
                    groupby = display_feature['groupby']
                    fig, axs = plt.subplots(nrows=1, ncols=groupby.ngroups)
                    for category, ax in zip(groupby.groups.items(), axs):
                        cat_data = data.iloc[category[1]]
                        ax.violinplot(cat_data['score'], showmedians=True)
                        ax.set_title(category[0])
                    plt.title(display_feature['feature'])
                    plt.show()
                elif display_feature['type'] == 'numeric':
                    plt.scatter(x=display_feature['feature'], y='score', data=data)
                    plt.title(display_feature['feature'])
                    plt.show()

        value = None

        if not display_params['features']:
            display = None

        return CheckResult(value, display=['explination of why...', display])
