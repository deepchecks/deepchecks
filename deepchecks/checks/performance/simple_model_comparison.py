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
import itertools
from typing import Callable, Dict, Hashable, List
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.dummy import DummyRegressor, DummyClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from deepchecks.checks.distribution.preprocessing import ScaledNumerics
from deepchecks.utils.strings import format_number

from deepchecks import CheckResult, Dataset
from deepchecks.base.check import ConditionResult, TrainTestBaseCheck
from deepchecks.utils.metrics import task_type_check, ModelType, initialize_multi_scorers, \
    get_scorers_list, get_scores_ratio
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
                + tree - runs a simple decision tree.
        alternative_scorers (Dict[str, Callable], default None):
            An optional dictionary of scorer name to scorer functions.
            If none given, using default scorers
        maximum_ratio (int):
            the ratio can be up to infinity so choose maximum value to limit to.
        max_depth (int):
            the max depth of the tree (used only if simple model type is tree).
        random_state (int):
            the random state (used only if simple model type is tree or random).
    """

    def __init__(self, simple_model_type: str = 'constant', alternative_scorers: Dict[str, Callable] = None,
                 maximum_ratio: int = 50, max_depth: int = 3, random_state: int = 42):
        super().__init__()
        self.simple_model_type = simple_model_type
        self.alternative_scorers = initialize_multi_scorers(alternative_scorers)
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
        Dataset.validate_dataset(train_dataset)
        Dataset.validate_dataset(test_dataset)
        train_dataset.validate_label()
        test_dataset.validate_label()
        validate_model(test_dataset, model)

        # Get default scorers if no alternative, or validate alternatives
        scorers = get_scorers_list(model, train_dataset, self.alternative_scorers, multiclass_avg=False)

        task_type = task_type_check(model, train_dataset)
        simple_model = self._create_simple_model(train_dataset, task_type)

        models = [
            (f'{type(model).__name__} model', 'Origin', model),
            (f'Simple model - {self.simple_model_type}', 'Simple', simple_model)
        ]
        if task_type == ModelType.MULTICLASS:
            results = []
            for model_name, model_type, model_instance in models:
                for scorer in scorers:
                    score_result = scorer(model_instance, test_dataset)
                    # Multiclass scorers return numpy array of result per class
                    for class_i, value in enumerate(score_result):
                        if scorer.is_negative_scorer():
                            display_value = -value
                        else:
                            display_value = value
                        results.append([model_name, model_type, value, display_value, scorer.name, class_i])
            # Figure
            results_df = pd.DataFrame(results, columns=['Model', 'Type', 'Value', 'DisplayVal', 'Metric', 'Class'])
            fig = px.bar(results_df, x=['Class', 'Model'], y='DisplayVal', color='Model', barmode='group',
                         facet_col='Metric', facet_col_spacing=0.05)
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
            fig.update_yaxes(title=None, matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        else:
            results = []
            for model_name, model_type, model_instance in models:
                for scorer in scorers:
                    score_result = scorer(model_instance, test_dataset)
                    if scorer.is_negative_scorer():
                        display_value = -score_result
                    else:
                        display_value = score_result
                    results.append([model_name, model_type, score_result, display_value, scorer.name])

            # Figure
            results_df = pd.DataFrame(results, columns=['Model', 'Type', 'Value', 'DisplayVal', 'Metric'])
            fig = px.bar(results_df, x='Model', y='DisplayVal', color='Model', barmode='group',
                         facet_col='Metric', facet_col_spacing=0.05)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

        return CheckResult({'scores': results_df, 'type': task_type}, display=fig)

    def _create_simple_model(self, train_ds: Dataset, task_type: ModelType):
        """Find the simple model score for given metric.

        Args:
            train_ds (Dataset): The training dataset object. Must contain an index.
            task_type (ModelType): the model type.
        Returns:
            score for simple and given model respectively and the score name in a tuple

        Raises:
            NotImplementedError: If the simple_model_type is not supported
        """
        np.random.seed(self.random_state)

        if self.simple_model_type == 'random':
            return RandomModel(train_ds.label_col)

        elif self.simple_model_type == 'constant':
            if task_type == ModelType.REGRESSION:
                clf = DummyRegressor(strategy='mean')
            elif task_type in {ModelType.BINARY, ModelType.MULTICLASS}:
                clf = DummyClassifier(strategy='most_frequent')
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')

            clf.fit(train_ds.features_columns, train_ds.label_col)
            return clf

        elif self.simple_model_type == 'tree':
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

            clf = Pipeline([('scaler', ScaledNumerics(train_ds.cat_features, max_num_categories=10)),
                            ('tree-model', clf)])
            clf.fit(train_ds.features_columns, train_ds.label_col)
            return clf
        else:
            raise DeepchecksValueError(
                f'Unknown model type - {self.simple_model_type}, expected to be one of '
                f"['random', 'constant', 'tree'] "
                f"but instead got {self.simple_model_type}"  # pylint: disable=inconsistent-quotes
            )

    def add_condition_ratio_not_less_than(self, min_allowed_ratio: float = 1.1, classes: List[Hashable] = None):
        """Add condition - require min allowed ratio between the given and the simple model.

        Args:
            min_allowed_ratio (float): Min allowed ratio between the given and the simple model -
            ratio is given model / simple model (if the scorer returns negative values we divide 1 by it)
            classes (List[Hashable]): Used in multiclass models to limit condition only to given classes.
        """
        def condition(result: Dict, max_ratio=self.maximum_ratio, class_list=classes) -> ConditionResult:
            scores_df = result['scores']
            task_type = result['type']
            metrics = scores_df['Metric'].unique()

            metrics_dfs = []
            if task_type == ModelType.MULTICLASS:
                if class_list is None:
                    class_list = scores_df['Class'].unique()
                for metric, clas in itertools.product(metrics, class_list):
                    score_rows = scores_df[(scores_df['Metric'] == metric) & (scores_df['Class'] == clas)]
                    metrics_dfs.append(score_rows)
            else:
                for metric in metrics:
                    score_rows = scores_df[(scores_df['Metric'] == metric)]
                    metrics_dfs.append(score_rows)

            not_passing_metrics = set()
            for df in metrics_dfs:
                origin_score = df[df['Type'] == 'Origin']['Value'].iloc[0]
                simple_score = df[df['Type'] == 'Simple']['Value'].iloc[0]
                ratio = get_scores_ratio(simple_score, origin_score, max_ratio)
                if ratio < min_allowed_ratio:
                    not_passing_metrics.update(df['Metric'].tolist())

            if not_passing_metrics:
                not_passing_metrics = sorted(not_passing_metrics)
                msg = f'Metrics with scores ratio lower than threshold: {", ".join(not_passing_metrics)}'
                return ConditionResult(False, msg)
            else:
                return ConditionResult(True)

        return self.add_condition(f'Ratio not less than {format_number(min_allowed_ratio)} '
                                  'between the given model\'s score and the simple model\'s score',
                                  condition)


class RandomModel:
    """Class used to randomly predict from given series of labels."""

    def __init__(self, labels):
        self.labels = labels

    def predict(self, a):
        return np.random.choice(self.labels, a.shape[0])

    def predict_proba(self, a):
        classes_num = self.labels.unique().shape[0]
        predictions = self.predict(a)

        def create_proba(x):
            proba = np.zeros(classes_num)
            proba[x] = 1
            return proba
        return np.apply_along_axis(create_proba, axis=1, arr=predictions)
