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
import warnings
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer, make_scorer, f1_score, precision_score, recall_score
from sklearn.base import ClassifierMixin, BaseEstimator


from deepchecks import base  # pylint: disable=unused-import; it is used for type annotations
from deepchecks import errors
from deepchecks.utils.strings import is_string_column
from deepchecks.utils.simple_models import PerfectModel
from deepchecks.utils.typing import BasicModel, ClassificationModel


__all__ = [
    'ModelType',
    'task_type_check',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_REGRESSION_SCORERS',
    'DEFAULT_BINARY_SCORERS',
    'DEFAULT_MULTICLASS_SCORERS',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'DeepcheckScorer',
    'get_gain',
    'init_validate_scorers',
    'get_default_scorers'
]


class ModelType(enum.Enum):
    """Enum containing supported task types."""

    REGRESSION = 'regression'
    BINARY = 'binary'
    MULTICLASS = 'multiclass'


DEFAULT_BINARY_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall':  make_scorer(recall_score, zero_division=0)
}


DEFAULT_MULTICLASS_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision - Macro Average': make_scorer(precision_score, average='macro', zero_division=0),
    'Recall - Macro Average': make_scorer(recall_score, average='macro', zero_division=0)
}

MULTICLASS_SCORERS_NON_AVERAGE = {
    'F1': make_scorer(f1_score, average=None, zero_division=0),
    'Precision': make_scorer(precision_score, average=None, zero_division=0),
    'Recall': make_scorer(recall_score, average=None, zero_division=0)
}


DEFAULT_REGRESSION_SCORERS = {
    'Neg RMSE': 'neg_root_mean_squared_error',
    'Neg MAE': 'neg_mean_absolute_error',
    'R2': 'r2'
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

    Parameters
    ----------
    scorer : t.Union[str, t.Callable]
        sklearn scorer name or callable
    name : str
        scorer name
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
        """Return data of dataset without null labels."""
        valid_idx = dataset.data[dataset.label_name].notna()
        return dataset.data[valid_idx]

    def _run_score(self, model, dataframe, dataset):
        return self.scorer(model, dataframe[dataset.features], dataframe[dataset.label_name])

    def __call__(self, model, dataset: 'base.Dataset'):
        """Run score with labels null filtering."""
        df = self.filter_nulls(dataset)
        return self._run_score(model, df, dataset)

    def score_perfect(self, dataset: 'base.Dataset'):
        """Calculate the perfect score of the current scorer for given dataset."""
        df = self.filter_nulls(dataset)
        perfect_model = PerfectModel()
        perfect_model.fit(None, df[dataset.label_name])
        score = self._run_score(perfect_model, df, dataset)
        if isinstance(score, np.ndarray):
            # We expect the perfect score to be equal for all the classes, so takes the first one
            first_score = score[0]
            if any(score != first_score):
                warnings.warn(f'Scorer {self.name} return different perfect score for differect classes')
            return first_score
        return score

    def validate_fitting(self, model, dataset: 'base.Dataset', should_return_array: bool):
        """Validate given scorer for the model and dataset."""
        df = self.filter_nulls(dataset)
        if should_return_array:
            # In order for scorer to return array in right length need to pass him samples from all labels
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
            # In order for scorer to run, it must have at least one sample of each label.
            single_label_data = df.groupby(dataset.label_name).head(1)
            result = self._run_score(model, single_label_data, dataset)
            if not isinstance(result, Number):
                raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return number '
                                                  f'but got: {type(result).__name__}')

    def is_negative_scorer(self):
        """If initialized as sklearn scorer name, return whether it's negative scorer."""
        return self.sklearn_scorer_name is not None and self.sklearn_scorer_name.startswith('neg_')


def task_type_check(
    model: BasicModel,
    dataset: 'base.Dataset'
) -> ModelType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Parameters
    ----------
    model : BasicModel
        Model object - used to check if it has predict_proba()
    dataset : base.Dataset
        dataset - used to count the number of unique labels

    Returns
    -------
    ModelType
        TaskType enum corresponding to the model and dataset
    """
    if dataset.label_name is None:
        raise errors.DatasetValidationError('Expected dataset with label')

    label_col = t.cast(pd.Series, dataset.data[dataset.label_name])
    if isinstance(model, BaseEstimator):
        if not hasattr(model, 'predict_proba'):
            if is_string_column(label_col):
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
            return (
                ModelType.MULTICLASS
                if label_col.nunique() > 2
                else ModelType.BINARY
            )
    elif isinstance(model, ClassificationModel):
        return (
            ModelType.MULTICLASS
            if label_col.nunique() > 2
            else ModelType.BINARY
        )
    else:
        return ModelType.REGRESSION


def get_default_scorers(model_type, class_avg: bool = True):
    """Get default scorers based on model type.

    Parameters
    ----------
    model_type : ModelType
        model type to return scorers for
    class_avg : bool, default True
        for classification whether to return scorers of average score or per class
    """
    return_array = model_type in [ModelType.MULTICLASS, ModelType.BINARY] and class_avg is False

    if return_array:
        return MULTICLASS_SCORERS_NON_AVERAGE

    else:
        return DEFAULT_SCORERS_DICT[model_type]


def init_validate_scorers(scorers: t.Mapping[str, t.Union[str, t.Callable]],
                          model: BasicModel,
                          dataset: 'base.Dataset',
                          class_avg: bool = True,
                          model_type=None) -> t.List[DeepcheckScorer]:
    """Initialize scorers and return all of them as deepchecks scorers.

    Parameters
    ----------
    scorers : Mapping[str, Union[str, Callable]]
        dict of scorers names to scorer sklearn_name/function
    model : BasicModel
        used to validate the scorers, and calculate mode_type if None.
    dataset : Dataset
        used to validate the scorers, and calculate mode_type if None.
    class_avg : bool , default True
        for classification whether to return scorers of average score or per class
    model_type : ModelType , default None
        model type to return scorers for
    """
    return_array = model_type in [ModelType.MULTICLASS, ModelType.BINARY] and class_avg is False
    scorers = [DeepcheckScorer(scorer, name) for name, scorer in scorers.items()]
    for s in scorers:
        s.validate_fitting(model, dataset, return_array)
    return scorers


def get_gain(base_score, score, perfect_score, max_gain):
    """Get gain between base score and score compared to the distance from the perfect score."""
    distance_from_perfect = perfect_score - base_score
    scores_diff = score - base_score
    if distance_from_perfect == 0:
        # If both base score and score are perfect, return 0 gain
        if scores_diff == 0:
            return 0
        # else base_score is better than score, return -max_gain
        return -max_gain
    ratio = scores_diff / distance_from_perfect
    if ratio < -max_gain:
        return -max_gain
    if ratio > max_gain:
        return max_gain
    return ratio
