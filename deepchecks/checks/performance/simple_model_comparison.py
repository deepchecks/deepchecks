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
import plotly.graph_objects as go
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from deepchecks.checks.distribution.preprocessing import preprocess_dataset_to_scaled_numerics
from deepchecks.utils.strings import format_number

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import ConditionResult, TrainTestBaseCheck
from deepchecks.utils.metrics import DEFAULT_SINGLE_SCORER, task_type_check, \
    ModelType, get_scores_ratio, get_scorer_single, initialize_single_scorer
from deepchecks.utils.validation import validate_model
from deepchecks.errors import DeepchecksValueError


__all__ = ['SimpleModelComparison']


class SimpleModelComparison(TrainTestBaseCheck):
    """Compare given model score to simple model score (according to given model type).

    Args:
        simple_model_type (str):
            Type of the simple model ['random', 'constant', 'tree'].
                + random - select one of the labels by random.
                + constant - in regression is mean value, in classification the most common value.
                + tree - runs a simple desion tree.
        scorer (Union[str, Callable]):
            Score to show, either function or sklearn scorer name.
            If is not given a default scorer (per the model type) will be used.
        maximum_ratio (int):
            the ratio can be up to infinity so choose maximum value to limit to.
        max_depth (int):
            the max depth of the tree (used only if simple model type is tree).
        random_state (int):
            the random state (used only if simple model type is tree or random).
    """

    def __init__(self, simple_model_type: str = 'constant', scorer: Union[str, Callable] = None,
                 maximum_ratio: int = 50, max_depth: int = 3, random_state: int = 42):
        super().__init__()
        self.simple_model_type = simple_model_type
        self.scorer = initialize_single_scorer(scorer)
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
                         ratio is given model / simple model (if the scorer returns negative values we divide 1 by it)
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
            score for simple and given model respectively and the score name in a tuple

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
            elif task_type in {ModelType.BINARY, ModelType.MULTICLASS}:
                counts = train_ds.label_col.mode()
                simple_pred = np.array([counts.index[0]] * len(test_df))
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')

        elif self.simple_model_type == 'tree':
            y_train = train_ds.label_col
            x_train, x_test = preprocess_dataset_to_scaled_numerics(
                baseline_features=train_ds.features_columns,
                test_features=test_ds.features_columns,
                categorical_columns=test_ds.cat_features,
                max_num_categories=10
            )

            if task_type == ModelType.REGRESSION:
                clf = DecisionTreeRegressor(
                    max_depth=self.max_depth,
                    random_state=self.random_state
                )
            elif task_type in {ModelType.BINARY, ModelType.MULTICLASS}:
                clf = DecisionTreeClassifier(
                    max_depth=self.max_depth,
                    random_state=self.random_state,
                    class_weight='balanced'
                )
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')

            clf = clf.fit(x_train, y_train)
            simple_pred = clf.predict(x_test)

        else:
            raise DeepchecksValueError(
                f'Unknown model type - {self.simple_model_type}, expected to be one of '
                f"['random', 'constant', 'tree'] "
                f"but instead got {self.simple_model_type}"  # pylint: disable=inconsistent-quotes
            )

        y_test = test_ds.label_col.values

        scorer = get_scorer_single(model, train_ds, self.scorer)

        simple_score = scorer(_DummyModel, Dataset(pd.DataFrame(simple_pred), label=y_test))
        pred_score = scorer(model, Dataset(test_ds.features_columns, label=y_test, cat_features=test_ds.cat_features))

        return simple_score, pred_score, scorer.name

    def _simple_model_comparison(self, train_dataset: Dataset, test_dataset: Dataset, model):
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        validate_model(test_dataset, model)

        task_type = task_type_check(model, train_dataset)
        simple_score, pred_score, score_name = self._find_score(train_dataset, test_dataset, task_type, model)

        if score_name == DEFAULT_SINGLE_SCORER[task_type]:
            score_name = str(score_name) + ' (Default)'

        ratio = get_scores_ratio(simple_score, pred_score, self.maximum_ratio)

        text = f'The given model performance is {_more_than_prefix_adder(ratio, self.maximum_ratio)} times the ' \
               f'performance of the simple model, measuring performance using the {score_name} metric.<br>' \
               f'{type(model).__name__} model prediction has achieved a score of {format_number(pred_score)} ' \
               f'compared to Simple {self.simple_model_type} prediction ' \
               f'which achieved a score of {format_number(simple_score)} on tested data.'

        models = [f'Simple model - {self.simple_model_type}', f'{type(model).__name__} model']
        results = [simple_score, pred_score]
        fig = go.Figure([go.Bar(x=models, y=results)])
        fig.update_layout(width=600, height=500)
        fig.update_yaxes(title=score_name)

        return CheckResult({'given_model_score': pred_score,
                            'simple_model_score': simple_score,
                            'ratio': ratio},
                           display=[text, fig])

    def add_condition_ratio_not_less_than(self, min_allowed_ratio: float = 1.1):
        """Add condition - require min allowed ratio between the given and the simple model.

        Args:
            min_allowed_ratio (float): Min allowed ratio between the given and the simple model -
            ratio is given model / simple model (if the scorer returns negative values we divide 1 by it)
        """
        def condition(result: Dict) -> ConditionResult:
            ratio = result['ratio']
            if ratio < min_allowed_ratio:
                return ConditionResult(False,
                                       f'The given model performs {_more_than_prefix_adder(ratio, self.maximum_ratio)} '
                                       'times compared to the simple model using the given scorer')
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ratio not less than {format_number(min_allowed_ratio)} '
                                  'between the given model\'s result and the simple model\'s result',
                                  condition)


class _DummyModel:
    @staticmethod
    def predict(a):
        return a

    @staticmethod
    def predict_proba(a):
        return a


def _more_than_prefix_adder(number, max_number):
    if number < max_number:
        return format_number(number)
    else:
        return 'more than ' + format_number(number)
