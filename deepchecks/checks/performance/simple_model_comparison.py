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
    get_scorers_list, get_scores_ratio, get_scorer_single
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

        # If user defined scorers used them, else use a single scorer
        if self.alternative_scorers:
            scorers = get_scorers_list(model, train_dataset, self.alternative_scorers, multiclass_avg=False)
        else:
            scorers = [get_scorer_single(model, train_dataset, multiclass_avg=False)]

        task_type = task_type_check(model, train_dataset)
        simple_model = self._create_simple_model(train_dataset, task_type)

        models = [
            (f'{type(model).__name__} model', 'Origin', model),
            (f'Simple model - {self.simple_model_type}', 'Simple', simple_model)
        ]

        # Multiclass have different return type from the scorer, list of score per class instead of single score
        if task_type == ModelType.MULTICLASS:
            results = []
            for model_name, model_type, model_instance in models:
                for scorer in scorers:
                    score_result: np.ndarray = scorer(model_instance, test_dataset)
                    # Multiclass scorers return numpy array of result per class
                    for class_i, class_score in enumerate(score_result):
                        # The proba returns in order of the sorted classes.
                        class_value = train_dataset.classes[class_i]
                        results.append([model_name, model_type, class_score, scorer.name, class_value])

            results_df = pd.DataFrame(results, columns=['Model', 'Type', 'Value', 'Metric', 'Class'])

            # Plot the metrics in a graph, grouping by the model and class
            fig = px.bar(results_df, x=['Class', 'Model'], y='Value', color='Model', barmode='group',
                         facet_col='Metric', facet_col_spacing=0.05)
            fig.update_xaxes(title=None, tickprefix='Class ', tickangle=60)
            fig.update_yaxes(title=None, matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))
        # Model is binary or regression
        else:
            results = []
            for model_name, model_type, model_instance in models:
                for scorer in scorers:
                    score_result: float = scorer(model_instance, test_dataset)
                    results.append([model_name, model_type, score_result, scorer.name])

            results_df = pd.DataFrame(results, columns=['Model', 'Type', 'Value', 'Metric'])

            # Plot the metrics in a graph, grouping by the model
            fig = px.bar(results_df, x='Model', y='Value', color='Model', barmode='group',
                         facet_col='Metric', facet_col_spacing=0.05)
            fig.update_xaxes(title=None)
            fig.update_yaxes(title=None, matches=None)
            fig.for_each_annotation(lambda a: a.update(text=a.text.split('=')[-1]))
            fig.for_each_yaxis(lambda yaxis: yaxis.update(showticklabels=True))

        return CheckResult({'scores': results_df, 'type': task_type}, display=fig)

    def _create_simple_model(self, train_ds: Dataset, task_type: ModelType):
        """Create a simple model of given type (random/constant/tree) to the given dataset.

        Args:
            train_ds (Dataset): The training dataset object.
            task_type (ModelType): the model type.
        Returns:
            Classifier object.

        Raises:
            NotImplementedError: If the simple_model_type is not supported
        """
        np.random.seed(self.random_state)

        if self.simple_model_type == 'random':
            simple_model = RandomModel()

        elif self.simple_model_type == 'constant':
            if task_type == ModelType.REGRESSION:
                simple_model = DummyRegressor(strategy='mean')
            elif task_type in {ModelType.BINARY, ModelType.MULTICLASS}:
                simple_model = DummyClassifier(strategy='most_frequent')
            else:
                raise DeepchecksValueError(f'Unknown task type - {task_type}')
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

            simple_model = Pipeline([('scaler', ScaledNumerics(train_ds.cat_features, max_num_categories=10)),
                                     ('tree-model', clf)])
        else:
            raise DeepchecksValueError(
                f'Unknown model type - {self.simple_model_type}, expected to be one of '
                f"['random', 'constant', 'tree'] "
                f"but instead got {self.simple_model_type}"  # pylint: disable=inconsistent-quotes
            )

        simple_model.fit(train_ds.features_columns, train_ds.label_col)
        return simple_model

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

            def get_ratio(df):
                simple_score = df[df['Type'] == 'Simple']['Value'].iloc[0]
                origin_score = df[df['Type'] == 'Origin']['Value'].iloc[0]
                return get_scores_ratio(simple_score, origin_score, max_ratio)

            fails = []
            if task_type == ModelType.MULTICLASS:
                if class_list is None:
                    class_list = scores_df['Class'].unique()
                for metric in metrics:
                    failed_classes = []
                    for clas in class_list:
                        score_rows = scores_df[(scores_df['Metric'] == metric) & (scores_df['Class'] == clas)]
                        ratio = get_ratio(score_rows)
                        if ratio < min_allowed_ratio:
                            failed_classes.append(str(clas))
                    if failed_classes:
                        fails.append(f'"{metric}" - Classes: {", ".join(failed_classes)}')
            else:
                for metric in metrics:
                    score_rows = scores_df[(scores_df['Metric'] == metric)]
                    ratio = get_ratio(score_rows)
                    if ratio < min_allowed_ratio:
                        fails.append(f'"{metric}"')

            if fails:
                msg = f'Metrics failed: {", ".join(sorted(fails))}'
                return ConditionResult(False, msg)
            else:
                return ConditionResult(True)

        return self.add_condition('$$\\frac{\\text{model score}}{\\text{simple model score}} >= '
                                  f'{format_number(min_allowed_ratio)}$$', condition)


class RandomModel:
    """Model used to randomly predict from given series of labels."""

    def __init__(self):
        self.labels = None

    def fit(self, X, y):  # pylint: disable=unused-argument,invalid-name
        # The X is not used, but it is needed to be matching to sklearn `fit` signature
        self.labels = y

    def predict(self, X):  # pylint: disable=invalid-name
        return np.random.choice(self.labels, X.shape[0])

    def predict_proba(self, X):  # pylint: disable=invalid-name
        classes = sorted(self.labels.unique().tolist())
        predictions = self.predict(X)

        def prediction_to_proba(y_pred):
            proba = np.zeros(len(classes))
            proba[classes.index(y_pred)] = 1
            return proba
        return np.apply_along_axis(prediction_to_proba, axis=1, arr=predictions)
