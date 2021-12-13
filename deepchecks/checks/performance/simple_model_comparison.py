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
"""Module containing simple comparison check."""
from typing import Callable, Dict, Union
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from deepchecks.checks.distribution.preprocessing import preprocess_dataset_to_scaled_numerics
from deepchecks.utils.strings import format_number

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import ConditionResult, TrainTestBaseCheck
from deepchecks.utils.metrics import DEFAULT_METRICS_DICT, DEFAULT_SINGLE_METRIC, task_type_check, \
                                     ModelType, validate_scorer, get_metrics_ratio
from deepchecks.utils.validation import validate_model

__all__ = ['SimpleModelComparison']


class DummyModel:
    @staticmethod
    def predict(a):
        return a

    @staticmethod
    def predict_proba(a):
        return a

def more_than_prefix_adder(number, max_number):
    if number < max_number:
        return format_number(number)
    else:
        return 'more than ' + format_number(number)

class SimpleModelComparison(TrainTestBaseCheck):
    """Compare given model score to simple model score (according to given model type).

    Args:
        simple_model_type (st):  Type of the simple model ['random', 'constant', 'tree'].
                    random - select one of the labels by random.
                    constant - in regression is mean value, in classification the most common value.
                    tree - runs a simple desion tree.
        metric (Union[str, Callable]): a custom metric given by user.
        metric_name (str): name of a default metric.
        maximum_ratio (int): the ratio can be up to infinity so choose maximum value to limit to.
        max_depth (int): the max depth of the tree (used only if simple model type is tree).
        random_state (int): the random state (used only if simple model type is tree or random).
    """

    def __init__(self, simple_model_type: str = 'constant', metric: Union[str, Callable] = None,
                 metric_name: str =None, maximum_ratio: int = 50, max_depth: int = 3, random_state: int = 42):
        super().__init__()
        self.simple_model_type = simple_model_type
        self.metric = metric
        self.metric_name = metric_name
        self.maximum_ratio = maximum_ratio
        self.max_depth = max_depth
        self.random_state = random_state

    def run(self, train_dataset, test_dataset, model) -> CheckResult:
        """Run check.

        Args:
            train_dataset (Dataset): The training dataset object. Must contain a label.
            test_dataset (Dataset): The test dataset object. Must contain a label.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.

        Returns:
            CheckResult: value is a Dict of: given_model_score, simple_model_score, ratio
                         ratio is given model / simple model (if the metric returns negative values we divide 1 by it)
                         if ratio is infinite max_ratio is returned

        Raises:
            DeepchecksValueError: If the object is not a Dataset instance.
        """
        return self._simple_model_comparison(train_dataset, test_dataset, model)

    def _find_score(self, train_ds: Dataset, test_ds: Dataset, task_type: ModelType, model):
        """Find the simple model score for given metric.

        Args:
            train_ds (Dataset): The training dataset object. Must contain an index.
            test_ds (Dataset): The test dataset object. Must contain an index.
            task_type (ModelType): the model type.
            model (BaseEstimator): A scikit-learn-compatible fitted estimator instance.
        Returns:
            score for simple and given model respectively and the metric type in a tuple

        Raises:
            NotImplementedError: If the simple_model_type is not supported

        """
        test_df = test_ds.data
        np.random.seed(self.random_state)

        if self.simple_model_type == 'random':
            simple_pred = np.random.choice(train_ds.label_col, test_df.shape[0])

        elif self.simple_model_type == 'constant':
            if task_type == ModelType.REGRESSION:
                simple_pred = np.array([np.mean(train_ds.label_col)] * len(test_df))

            elif task_type in (ModelType.BINARY, ModelType.MULTICLASS):
                counts = train_ds.label_col.mode()
                simple_pred = np.array([counts.index[0]] * len(test_df))

        elif self.simple_model_type == 'tree':
            y_train = train_ds.label_col
            x_train, x_test = preprocess_dataset_to_scaled_numerics(
                baseline_features= train_ds.features_columns,
                test_features=test_ds.features_columns,
                categorical_columns=test_ds.cat_features,
                max_num_categories=10
            )

            if task_type == ModelType.REGRESSION:
                clf = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state)
            elif task_type in (ModelType.BINARY, ModelType.MULTICLASS):
                clf = DecisionTreeClassifier(max_depth=self.max_depth,
                                             random_state=self.random_state, class_weight='balanced')
            if clf:
                clf = clf.fit(x_train, y_train)
                simple_pred = clf.predict(x_test)

        else:
            raise NotImplementedError(f"expected to be one of ['random', 'constant', 'tree'] \
                                    but instead got {self.simple_model_type}")

        y_test = test_ds.label_col

        if self.metric is not None:
            scorer = validate_scorer(self.metric, model, train_ds)
            metric_name = self.metric_name or self.metric if isinstance(self.metric, str) else 'User metric'
        else:
            metric_name = DEFAULT_SINGLE_METRIC[task_type]
            scorer = DEFAULT_METRICS_DICT[task_type][metric_name]

        simple_metric = scorer(DummyModel, simple_pred, y_test)
        pred_metric = scorer(model, test_ds.features_columns, y_test)

        return simple_metric, pred_metric, metric_name

    def _simple_model_comparison(self, train_dataset: Dataset, test_dataset: Dataset, model):
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        validate_model(test_dataset, model)

        simple_metric, pred_metric, metric_name = self._find_score(train_dataset, test_dataset,
                                                            task_type_check(model, train_dataset), model)

        ratio = get_metrics_ratio(simple_metric, pred_metric, self.maximum_ratio)

        text = f'The given model performs {more_than_prefix_adder(ratio, self.maximum_ratio)} times compared to' \
               f' the simple model using the {metric_name} metric.<br>' \
               f'{type(model).__name__} model prediction has achieved a score of {format_number(pred_metric)} ' \
               f'compared to Simple {self.simple_model_type} prediction ' \
               f'which achieved a score of {format_number(simple_metric)} on tested data.'

        def display_func():
            fig = plt.figure()
            ax = fig.add_axes([0, 0, 1, 1])
            models = [f'Simple model - {self.simple_model_type}', f'{type(model).__name__} model']
            metrics_results = [simple_metric, pred_metric]
            ax.bar(models, metrics_results)
            ax.set_ylabel(metric_name)

        return CheckResult({'given_model_score': pred_metric,
                            'simple_model_score': simple_metric,
                            'ratio': ratio},
                           display=[text, display_func])

    def add_condition_ratio_not_less_than(self, min_allowed_ratio: float = 1.1):
        """Add condition - require min allowed ratio between the given and the simple model.

        Args:
            min_allowed_ratio (float): Min allowed ratio between the given and the simple model -
            ratio is given model / simple model (if the metric returns negative values we divide 1 by it)
        """
        def condition(result: Dict) -> ConditionResult:
            ratio = result['ratio']
            if ratio < min_allowed_ratio:
                return ConditionResult(False,
                                       f'The given model performs {more_than_prefix_adder(ratio, self.maximum_ratio)}'
                                       f' times compared to the simple model using the given metric')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ratio not less than {format_number(min_allowed_ratio)} '
                                  f'between the given model\'s result and the simple model\'s result',
                                  condition)
