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
"""Utils module containing utilities for checks working with metrics."""
import typing as t
import enum
from numbers import Number

import numpy as np
import pandas as pd

from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score
from sklearn.base import ClassifierMixin, RegressorMixin


from deepchecks import base  # pylint: disable=unused-import; it is used for type annotations
from deepchecks import errors
from deepchecks.utils import validation


__all__ = [
    'ModelType',
    'task_type_check',
    'get_scorers_list',
    'initialize_single_scorer',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_SINGLE_SCORER',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'get_scores_ratio',
    'initialize_multi_scorers',
    'get_scorer_single'
]

from deepchecks.utils.strings import is_string_column


class ModelType(enum.Enum):
    """Enum containing suppoerted task types."""

    REGRESSION = 'regression'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'


DEFAULT_BINARY_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall'
}


DEFAULT_MULTICLASS_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision - Macro Average': 'precision_macro',
    'Recall - Macro Average': 'recall_macro'
}

MULTICLASS_SCORERS_NON_AVERAGE = {
    'F1': make_scorer(f1_score, average=None),
    'Precision': make_scorer(precision_score, average=None),
    'Recall': make_scorer(recall_score, average=None)
}


DEFAULT_REGRESSION_SCORERS = {
    'RMSE': 'neg_root_mean_squared_error',
    'MAE': 'neg_mean_absolute_error'
}


DEFAULT_SINGLE_SCORER = {
    ModelType.BINARY: 'Accuracy',
    ModelType.MULTICLASS: 'Accuracy',
    ModelType.REGRESSION: 'RMSE'
}


DEFAULT_SCORERS_DICT = {
    ModelType.BINARY: DEFAULT_BINARY_SCORERS,
    ModelType.MULTICLASS: DEFAULT_MULTICLASS_SCORERS,
    ModelType.REGRESSION: DEFAULT_REGRESSION_SCORERS
}


class DeepcheckScorer:
    """Encapsulate scorer function with extra methods.

    Scorer functions are functions used to compute various performance metrics, using the model and data as inputs,
    rather than the labels and predictions. Scorers are callables with the signature scorer(model, features, y_true).
    Additional data on scorer functions can be found at https://scikit-learn.org/stable/modules/model_evaluation.html.

    Args:
        scorer (t.Union[str, t.Callable]): sklearn scorer name or callable
        name (str): scorer name
    """

    def __init__(self, scorer: t.Union[str, t.Callable], name: str):
        self.name = name
        if isinstance(scorer, str):
            self.scorer: t.Callable = get_scorer(scorer)
            self.sklearn_scorer_name = scorer
        elif callable(scorer):
            self.scorer: t.Callable = scorer
            self.sklearn_scorer_name = None
        else:
            scorer_type = type(scorer).__name__
            if name:
                message = f'Scorer {name} value should be either a callable or string but got: {scorer_type}'
            else:
                message = f'Scorer should be should be either a callable or string but got: {scorer_type}'
            raise errors.DeepchecksValueError(message)

    @classmethod
    def filter_nulls(cls, dataset: 'base.Dataset'):
        valid_idx = dataset.label_col.notna()
        return dataset.data[valid_idx]

    def _run_score(self, model, dataframe, dataset):
        return self.scorer(model, dataframe[dataset.features], dataframe[dataset.label_name])

    def __call__(self, model, dataset: 'base.Dataset'):
        df = self.filter_nulls(dataset)
        return self._run_score(model, df, dataset)

    def validate_fitting(self, model, dataset: 'base.Dataset', should_return_array: bool):
        """Validate given scorer for the model and dataset."""
        df = self.filter_nulls(dataset)
        if should_return_array:
            # In order for multiclass scorer to return array in right length need to pass him samples from all labels
            single_label_data = df.groupby(dataset.label_name).head(1)
            result = self._run_score(model, single_label_data, dataset)
            if not isinstance(result, np.ndarray):
                raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return np.ndarray '
                                                  f'but got: {type(result).__name__}')
            expected_types = t.cast(
                str,
                np.typecodes['AllInteger'] + np.typecodes['AllFloat']  # type: ignore
            )
            kind = result.dtype.kind
            if kind not in expected_types:
                raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return np.ndarray of number kind '
                                                  f'but got: {kind}')
            # Validate returns value for each class
            if len(result) != len(single_label_data):
                raise errors.DeepchecksValueError(f'Found {len(single_label_data)} classes, but scorer {self.name} '
                                                  f'returned only {len(result)} elements in the score array value')

        else:
            result = self._run_score(model, df.head(2), dataset)
            if not isinstance(result, Number):
                raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return number '
                                                  f'but got: {type(result).__name__}')

    def is_negative_scorer(self):
        return self.sklearn_scorer_name is not None and self.sklearn_scorer_name.startswith('neg_')


def task_type_check(
    model: t.Union[ClassifierMixin, RegressorMixin],
    dataset: 'base.Dataset'
) -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels

    Returns:
        TaskType enum corresponding to the model and dataset
    """
    validation.model_type_validation(model)
    dataset.validate_label()

    if not hasattr(model, 'predict_proba'):
        if is_string_column(dataset.label_col):
            raise errors.DeepchecksValueError(
                'Model was identified as a regression model, but label column was found to contain strings.'
            )
        elif isinstance(model, ClassifierMixin):
            raise errors.DeepchecksValueError(
                'Model is a sklearn classification model (a subclass of ClassifierMixin), but lacks the '
                'predict_proba method. Please train the model with probability=True, or skip / ignore this check.'
            )
        else:
            return ModelType.REGRESSION
    else:
        labels = t.cast(pd.Series, dataset.label_col)

        return (
            ModelType.MULTICLASS
            if labels.nunique() > 2
            else ModelType.BINARY
        )


def task_type_validation(
    model: t.Union[ClassifierMixin, RegressorMixin],
    dataset: 'base.Dataset',
    expected_types: t.Sequence[ModelType]
):
    """Validate task type (regression, binary, multiclass) according to model object and label column.

    Args:
        model (Union[ClassifierMixin, RegressorMixin]): Model object - used to check if has predict_proba()
        dataset (Dataset): dataset - used to count the number of unique labels
        expected_types (List[ModelType]): allowed types of model
        check_name (str): check name to print in error

    Raises:
            DeepchecksValueError if model type doesn't match one of the expected_types
    """
    task_type = task_type_check(model, dataset)
    if task_type not in expected_types:
        raise errors.DeepchecksValueError(
            f'Expected model to be a type from {[e.value for e in expected_types]}, '
            f'but received model of type: {task_type.value}'
        )


def get_scorers_list(
    model,
    dataset: 'base.Dataset',
    alternative_scorers: t.Dict[str, t.Callable] = None,
    multiclass_avg: bool = True
) -> t.List[DeepcheckScorer]:
    """Return list of scorer objects to use in a score-dependant check.

    If no alternative_scorers is supplied, then a default list of scorers is used per task type, as it is inferred
    from the dataset and model. If a list is supplied, then the scorer functions are checked and used instead.

    Args:
        model (BaseEstimator): Model object for which the scores would be calculated
        dataset (Dataset): Dataset object on which the scores would be calculated
        alternative_scorers (Dict[str, Callable]): Optional dictionary of sklearn scorers to use instead of default list
        multiclass_avg

    Returns:
        Dictionary containing names of scorer functions.
    """
    # Check for model type
    model_type = task_type_check(model, dataset)
    multiclass_array = model_type == ModelType.MULTICLASS and multiclass_avg is False

    if alternative_scorers:
        scorers = alternative_scorers
    else:
        if multiclass_array:
            default_scorers = MULTICLASS_SCORERS_NON_AVERAGE
        else:
            default_scorers = DEFAULT_SCORERS_DICT[model_type]
        # Transform dict of scorers to deepchecks' scorers
        scorers = initialize_multi_scorers(default_scorers)

    for s in scorers:
        s.validate_fitting(model, dataset, multiclass_array)

    return scorers


def get_scorer_single(model, dataset: 'base.Dataset', alternative_scorer: t.Optional[DeepcheckScorer] = None,
                      multiclass_avg: bool = True):
    """Return single score to use in check, and validate scorer fit the model and dataset."""
    model_type = task_type_check(model, dataset)
    multiclass_array = model_type == ModelType.MULTICLASS and multiclass_avg is False

    if alternative_scorer is None:
        if multiclass_array:
            scorer_name = 'F1'
            scorer_func = MULTICLASS_SCORERS_NON_AVERAGE[scorer_name]
        else:
            scorer_name = DEFAULT_SINGLE_SCORER[model_type]
            scorer_func = DEFAULT_SCORERS_DICT[model_type][scorer_name]
        alternative_scorer = DeepcheckScorer(scorer_func, scorer_name)

    alternative_scorer.validate_fitting(model, dataset, multiclass_array)
    return alternative_scorer


def initialize_single_scorer(scorer: t.Optional[t.Union[str, t.Callable]], scorer_name=None) \
        -> t.Optional[DeepcheckScorer]:
    """If string, get scorer from sklearn. If none, return none."""
    if scorer is None:
        return None

    scorer_name = scorer_name or (scorer if isinstance(scorer, str) else 'User Scorer')
    return DeepcheckScorer(scorer, scorer_name)


def initialize_multi_scorers(alternative_scorers: t.Optional[t.Mapping[str, t.Callable]]) -> \
        t.Optional[t.List[DeepcheckScorer]]:
    """Initialize user scorers and return all of them as callable."""
    if alternative_scorers is None:
        return None
    elif len(alternative_scorers) == 0:
        raise errors.DeepchecksValueError('Scorers dictionary can\'t be empty')
    else:
        return [DeepcheckScorer(scorer, name) for name, scorer in alternative_scorers.items()]


def get_scores_ratio(train_score: float, test_score: float, max_ratio=np.Inf) -> float:
    """Return the ratio of test metric compared to train metric."""
    if train_score == 0:
        return max_ratio
    else:
        ratio = test_score / train_score
        if train_score < 0 and test_score < 0:
            ratio = 1 / ratio
        ratio = min(max_ratio, ratio)
        return ratio
