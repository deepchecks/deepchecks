# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Utils module containing utilities for checks working with scorers."""
import typing as t
from numbers import Number

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import f1_score, get_scorer, make_scorer, precision_score, recall_score
from sklearn.metrics._scorer import _BaseScorer, _ProbaScorer

from deepchecks import tabular  # pylint: disable=unused-import; it is used for type annotations
from deepchecks.core import errors
from deepchecks.tabular.metric_utils.additional_classification_metrics import (false_negative_rate_metric,
                                                                               false_positive_rate_metric,
                                                                               true_negative_rate_metric)
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.logger import get_logger
from deepchecks.utils.metrics import get_scorer_name
from deepchecks.utils.simple_models import PerfectModel
from deepchecks.utils.strings import is_string_column
from deepchecks.utils.typing import BasicModel, ClassificationModel

__all__ = [
    'task_type_check',
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_REGRESSION_SCORERS',
    'DEFAULT_BINARY_SCORERS',
    'DEFAULT_MULTICLASS_SCORERS',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'DeepcheckScorer',
    'init_validate_scorers',
    'get_default_scorers'
]


DEFAULT_BINARY_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision': make_scorer(precision_score, zero_division=0),
    'Recall': make_scorer(recall_score, zero_division=0)
}

DEFAULT_MULTICLASS_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision - Macro Average': make_scorer(precision_score, average='macro', zero_division=0),
    'Recall - Macro Average': make_scorer(recall_score, average='macro', zero_division=0),
}

MULTICLASS_SCORERS_NON_AVERAGE = {
    'F1': 'f1_per_class',
    'Precision': 'precision_per_class',
    'Recall': 'recall_per_class',
}

DEFAULT_REGRESSION_SCORERS = {
    'Neg RMSE': 'neg_root_mean_squared_error',
    'Neg MAE': 'neg_mean_absolute_error',
    'R2': 'r2',
}

DEFAULT_SCORERS_DICT = {
    TaskType.BINARY: DEFAULT_BINARY_SCORERS,
    TaskType.MULTICLASS: DEFAULT_MULTICLASS_SCORERS,
    TaskType.REGRESSION: DEFAULT_REGRESSION_SCORERS
}


_func_dict = {
    'neg_rmse': 'neg_root_mean_squared_error',
    'neg_mae': 'neg_mean_absolute_error',
    'precision_per_class': make_scorer(precision_score, average=None, zero_division=0),
    'recall_per_class': make_scorer(recall_score, average=None, zero_division=0),
    'f1_per_class': make_scorer(f1_score, average=None, zero_division=0),
    'fpr_per_class': make_scorer(false_positive_rate_metric, averaging_method='per_class'),
    'fpr': make_scorer(false_positive_rate_metric, averaging_method='binary'),
    'fpr_macro': make_scorer(false_positive_rate_metric, averaging_method='macro'),
    'fpr_micro': make_scorer(false_positive_rate_metric, averaging_method='micro'),
    'fpr_weighted': make_scorer(false_positive_rate_metric, averaging_method='weighted'),
    'fnr_per_class': make_scorer(false_negative_rate_metric, averaging_method='per_class'),
    'fnr': make_scorer(false_negative_rate_metric, averaging_method='binary'),
    'fnr_macro': make_scorer(false_negative_rate_metric, averaging_method='macro'),
    'fnr_micro': make_scorer(false_negative_rate_metric, averaging_method='micro'),
    'fnr_weighted': make_scorer(false_negative_rate_metric, averaging_method='weighted'),
    'tnr_per_class': make_scorer(true_negative_rate_metric, averaging_method='per_class'),
    'tnr': make_scorer(true_negative_rate_metric, averaging_method='binary'),
    'tnr_macro': make_scorer(true_negative_rate_metric, averaging_method='macro'),
    'tnr_micro': make_scorer(true_negative_rate_metric, averaging_method='micro'),
    'tnr_weighted': make_scorer(true_negative_rate_metric, averaging_method='weighted'),
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

    def __init__(self, scorer: t.Union[str, t.Callable], name: str = None):
        if isinstance(scorer, str):
            formated_scorer_name = scorer.lower().replace('sensitivity', 'recall').replace('specificity', 'tnr')\
                .replace(' ', '_')
            if formated_scorer_name in _func_dict:
                self.scorer = _func_dict[formated_scorer_name]
            else:
                try:
                    self.scorer = get_scorer(scorer)
                except ValueError as e:
                    raise errors.DeepchecksValueError(f'Scorer name {scorer} is unknown. '
                                                      f'See metric guide for a list of allowed scorer names.') from e
        elif callable(scorer):
            self.scorer = scorer
        else:
            scorer_type = type(scorer).__name__
            msg = f'Scorer {name if name else ""} value should be either a callable or string but got: {scorer_type}'
            raise errors.DeepchecksValueError(msg)
        self.name = name if name else get_scorer_name(scorer)

    @classmethod
    def filter_nulls(cls, dataset: 'tabular.Dataset') -> 'tabular.Dataset':
        """Return data of dataset without null labels."""
        valid_idx = dataset.data[dataset.label_name].notna()
        return dataset.copy(dataset.data[valid_idx])

    def run_on_data_and_label(self, model, data: pd.DataFrame, label_col):
        """Run scorer with model, data and labels without null filtering."""
        return self.scorer(model, data, label_col)

    def run_on_pred(self, y_true, y_pred=None, y_proba=None):
        """Run sklearn scorer on the labels and the pred/proba according to scorer type."""
        # pylint: disable=protected-access
        if isinstance(self.scorer, _BaseScorer):
            if y_proba and isinstance(self.scorer, _ProbaScorer):
                pred_to_use = y_proba
            else:
                pred_to_use = y_pred
            return self.scorer._score_func(y_true, pred_to_use, **self.scorer._kwargs) * self.scorer._sign
        raise errors.DeepchecksValueError('Only supports sklearn scorers')

    def _run_score(self, model, dataset: 'tabular.Dataset'):
        return self.scorer(model, dataset.features_columns, dataset.label_col)

    def __call__(self, model, dataset: 'tabular.Dataset'):
        """Run score with labels null filtering."""
        return self._run_score(model, self.filter_nulls(dataset))

    def score_perfect(self, dataset: 'tabular.Dataset'):
        """Calculate the perfect score of the current scorer for given dataset."""
        dataset = self.filter_nulls(dataset)
        perfect_model = PerfectModel()
        perfect_model.fit(None, dataset.label_col)
        score = self._run_score(perfect_model, dataset)
        if isinstance(score, np.ndarray):
            # We expect the perfect score to be equal for all the classes, so takes the first one
            first_score = score[0]
            if any(score != first_score):
                get_logger().warning('Scorer %s return different perfect score for differect classes', self.name)
            return first_score
        return score

    def validate_fitting(self, model, dataset: 'tabular.Dataset'):
        """Validate given scorer for the model and dataset."""
        dataset.assert_features()
        dataset = self.filter_nulls(dataset)
        # In order for scorer to return result in right dimensions need to pass it samples from all labels
        single_label_data = dataset.data.groupby(dataset.label_name).head(1)
        result = self._run_score(model, dataset.copy(single_label_data))

        if isinstance(result, np.ndarray):
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
        elif not isinstance(result, Number):
            raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return number or np.ndarray '
                                              f'but got: {type(result).__name__}')


def task_type_check(
        model: BasicModel,
        dataset: 'tabular.Dataset'
) -> TaskType:
    """Check task type (regression, binary, multiclass) according to model object and label column.

    Parameters
    ----------
    model : BasicModel
        Model object - used to check if it has predict_proba()
    dataset : tabular.Dataset
        dataset - used to count the number of unique labels

    Returns
    -------
    TaskType
        TaskType enum corresponding to the model and dataset
    """
    label_col = dataset.label_col
    if not model:
        return dataset.label_type
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
                return TaskType.REGRESSION
        else:
            return (
                TaskType.MULTICLASS
                if label_col.nunique() > 2
                else TaskType.BINARY
            )
    if isinstance(model, ClassificationModel):
        return (
            TaskType.MULTICLASS
            if label_col.nunique() > 2
            else TaskType.BINARY
        )
    return TaskType.REGRESSION


def get_default_scorers(model_type, class_avg: bool = True):
    """Get default scorers based on model type.

    Parameters
    ----------
    model_type : TaskType
        model type to return scorers for
    class_avg : bool, default True
        for classification whether to return scorers of average score or per class
    """
    return_array = model_type in [TaskType.MULTICLASS, TaskType.BINARY] and class_avg is False

    if return_array:
        return MULTICLASS_SCORERS_NON_AVERAGE
    else:
        return DEFAULT_SCORERS_DICT[model_type]


def init_validate_scorers(scorers: t.Union[t.Mapping[str, t.Union[str, t.Callable]], t.List[str]],
                          model: BasicModel,
                          dataset: 'tabular.Dataset') -> t.List[DeepcheckScorer]:
    """Initialize scorers and return all of them as deepchecks scorers.

    Parameters
    ----------
    scorers : Mapping[str, Union[str, Callable]]
        dict of scorers names to scorer sklearn_name/function or a list without a name
    model : BasicModel
        used to validate the scorers, and calculate mode_type if None.
    dataset : Dataset
        used to validate the scorers, and calculate mode_type if None.
    """
    if isinstance(scorers, t.Mapping):
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer, name) for name, scorer in scorers.items()]
    else:
        scorers: t.List[DeepcheckScorer] = [DeepcheckScorer(scorer) for scorer in scorers]
    for s in scorers:
        s.validate_fitting(model, dataset)
    return scorers
