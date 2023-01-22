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
import logging
import typing as t
import warnings
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.metrics import get_scorer, log_loss, make_scorer, mean_absolute_error, mean_squared_error
from sklearn.metrics._scorer import _BaseScorer, _ProbaScorer
from sklearn.preprocessing import LabelBinarizer

try:
    from deepchecks_metrics import f1_score, jaccard_score, precision_score, recall_score  # noqa: F401
except ImportError:
    from sklearn.metrics import \
        f1_score, recall_score, precision_score, jaccard_score  # noqa: F401  pylint: disable=ungrouped-imports

from deepchecks.core import errors
from deepchecks.tabular.metric_utils.additional_classification_metrics import (false_negative_rate_metric,
                                                                               false_positive_rate_metric,
                                                                               roc_auc_per_class,
                                                                               true_negative_rate_metric)
from deepchecks.tabular.utils.task_type import TaskType
from deepchecks.utils.docref import doclink
from deepchecks.utils.logger import get_logger
from deepchecks.utils.metrics import get_scorer_name
from deepchecks.utils.simple_models import PerfectModel
from deepchecks.utils.typing import BasicModel

if TYPE_CHECKING:
    from deepchecks import tabular  # pylint: disable=unused-import; it is used for type annotations

__all__ = [
    'DEFAULT_SCORERS_DICT',
    'DEFAULT_REGRESSION_SCORERS',
    'DEFAULT_BINARY_SCORERS',
    'DEFAULT_MULTICLASS_SCORERS',
    'MULTICLASS_SCORERS_NON_AVERAGE',
    'DeepcheckScorer',
    'init_validate_scorers',
    'get_default_scorers',
    'regression_scorers_lower_is_better_dict',
    'regression_scorers_higher_is_better_dict',
    'binary_scorers_dict',
    'multiclass_scorers_dict',
    'validate_proba'
]

DEFAULT_BINARY_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision': 'precision',
    'Recall': 'recall',
}

DEFAULT_MULTICLASS_SCORERS = {
    'Accuracy': 'accuracy',
    'Precision - Macro Average': 'precision_macro',
    'Recall - Macro Average': 'recall_macro',
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

regression_scorers_lower_is_better_dict = {
    'mean_squared_error': make_scorer(mean_squared_error, squared=True),
    'mse': make_scorer(mean_squared_error, squared=True),
    'root_mean_squared_error': make_scorer(mean_squared_error, squared=False),
    'rmse': make_scorer(mean_squared_error, squared=False),
    'mae': make_scorer(mean_absolute_error),
    'mean_absolute_error': make_scorer(mean_absolute_error),
}

regression_scorers_higher_is_better_dict = {
    'neg_mse': get_scorer('neg_mean_squared_error'),
    'neg_rmse': get_scorer('neg_root_mean_squared_error'),
    'neg_mae': get_scorer('neg_mean_absolute_error'),
    'r2': get_scorer('r2'),
    'explained_variance': get_scorer('explained_variance'),
    'max_error': get_scorer('max_error'),
    'neg_mean_squared_log_error': get_scorer('neg_mean_squared_log_error'),
    'neg_median_absolute_error': get_scorer('neg_median_absolute_error'),
    'neg_mean_poisson_deviance': get_scorer('neg_mean_poisson_deviance'),
    'neg_mean_gamma_deviance': get_scorer('neg_mean_gamma_deviance')
}

binary_scorers_dict = {
    'accuracy': get_scorer('accuracy'),
    'balanced_accuracy': get_scorer('balanced_accuracy'),
    'average_precision': get_scorer('average_precision'),
    'neg_brier_score': get_scorer('neg_brier_score'),
    'precision': make_scorer(precision_score, zero_division=0),
    'recall': make_scorer(recall_score, zero_division=0),
    'fpr': make_scorer(false_positive_rate_metric, averaging_method='binary'),
    'fnr': make_scorer(false_negative_rate_metric, averaging_method='binary'),
    'tnr': make_scorer(true_negative_rate_metric, averaging_method='binary'),
    'jaccard': make_scorer(jaccard_score, zero_division=0),
    'roc_auc': get_scorer('roc_auc'),
    'neg_log_loss': make_scorer(log_loss, greater_is_better=False, needs_proba=True, labels=[0, 1])
}

multiclass_scorers_dict = {
    'accuracy': get_scorer('accuracy'),
    'precision_macro': make_scorer(precision_score, average='macro', zero_division=0),
    'precision_micro': make_scorer(precision_score, average='micro', zero_division=0),
    'precision_weighted': make_scorer(precision_score, average='weighted', zero_division=0),
    'precision_per_class': make_scorer(precision_score, average=None, zero_division=0),
    'recall_macro': make_scorer(recall_score, average='macro', zero_division=0),
    'recall_micro': make_scorer(recall_score, average='micro', zero_division=0),
    'recall_weighted': make_scorer(recall_score, average='weighted', zero_division=0),
    'recall_per_class': make_scorer(recall_score, average=None, zero_division=0),
    'f1_macro': make_scorer(f1_score, average='macro', zero_division=0),
    'f1_micro': make_scorer(f1_score, average='micro', zero_division=0),
    'f1_weighted': make_scorer(f1_score, average='weighted', zero_division=0),
    'f1_per_class': make_scorer(f1_score, average=None, zero_division=0),
    'fpr_per_class': make_scorer(false_positive_rate_metric, averaging_method='per_class'),
    'fpr_macro': make_scorer(false_positive_rate_metric, averaging_method='macro'),
    'fpr_micro': make_scorer(false_positive_rate_metric, averaging_method='micro'),
    'fpr_weighted': make_scorer(false_positive_rate_metric, averaging_method='weighted'),
    'fnr_per_class': make_scorer(false_negative_rate_metric, averaging_method='per_class'),
    'fnr_macro': make_scorer(false_negative_rate_metric, averaging_method='macro'),
    'fnr_micro': make_scorer(false_negative_rate_metric, averaging_method='micro'),
    'fnr_weighted': make_scorer(false_negative_rate_metric, averaging_method='weighted'),
    'tnr_per_class': make_scorer(true_negative_rate_metric, averaging_method='per_class'),
    'tnr_macro': make_scorer(true_negative_rate_metric, averaging_method='macro'),
    'tnr_micro': make_scorer(true_negative_rate_metric, averaging_method='micro'),
    'tnr_weighted': make_scorer(true_negative_rate_metric, averaging_method='weighted'),
    'roc_auc_per_class': make_scorer(roc_auc_per_class, needs_proba=True),
    'roc_auc_ovr': get_scorer('roc_auc_ovr'),
    'roc_auc_ovo': get_scorer('roc_auc_ovo'),
    'roc_auc_ovr_weighted': get_scorer('roc_auc_ovr_weighted'),
    'roc_auc_ovo_weighted': get_scorer('roc_auc_ovo_weighted'),
    'jaccard_macro': make_scorer(jaccard_score, average='macro', zero_division=0),
    'jaccard_micro': make_scorer(jaccard_score, average='micro', zero_division=0),
    'jaccard_weighted': make_scorer(jaccard_score, average='weighted', zero_division=0),
    'jaccard_per_class': make_scorer(jaccard_score, average=None, zero_division=0),
}

_str_to_scorer_dict = {**regression_scorers_higher_is_better_dict,
                       **regression_scorers_lower_is_better_dict,
                       **multiclass_scorers_dict,
                       **binary_scorers_dict}

SUPPORTED_MODELS_DOCLINK = doclink('supported-prediction-format',
                                   template='For more information please refer to the Supported Models guide {link}')


class DeepcheckScorer:
    """Encapsulate scorer function with extra methods.

    Scorer functions are functions used to compute various performance metrics, using the model and data as inputs,
    rather than the labels and predictions. Scorers are callables with the signature scorer(model, features, y_true).
    Additional data on scorer functions can be found at https://scikit-learn.org/stable/modules/model_evaluation.html.

    Parameters
    ----------
    scorer : t.Union[str, t.Callable]
        sklearn scorer name or callable
    model_classes: t.Optional[t.List]
        possible classes output for model. None for regression tasks.
    name : str, default = None
        scorer name
    """

    def __init__(self,
                 scorer: t.Union[str, t.Callable],
                 model_classes: t.Optional[t.List],
                 observed_classes: t.Optional[t.List],
                 name: str = None):
        if isinstance(scorer, str):
            formatted_scorer_name = scorer.lower().replace('sensitivity', 'recall').replace('specificity', 'tnr') \
                .replace(' ', '_')
            if formatted_scorer_name in regression_scorers_lower_is_better_dict:
                warnings.warn(f'Deepchecks checks assume higher metric values represent better performance. '
                              f'{formatted_scorer_name} does not follow that convention.')
            if formatted_scorer_name in _str_to_scorer_dict:
                self.scorer = _str_to_scorer_dict[formatted_scorer_name]
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
        self.model_classes = model_classes
        self.observed_classes = observed_classes

    @classmethod
    def filter_nulls(cls, dataset: 'tabular.Dataset') -> 'tabular.Dataset':
        """Return data of dataset without null labels."""
        valid_idx = dataset.data[dataset.label_name].notna()
        return dataset.copy(dataset.data[valid_idx])

    def run_on_data_and_label(self, model, data: pd.DataFrame, label_col):
        """Run scorer with model, data and labels without null filtering."""
        return self._run_score(model, data, label_col)

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

    def _wrap_classification_model(self, model):
        """Convert labels to 0/1 if model is a binary classifier, and converts to multi-label if multiclass."""

        class MyModelWrapper:
            """Convert labels to 0/1 if model is a binary classifier, and converts to multi-label if multiclass."""

            def __init__(self, user_model, model_classes):
                self.user_model = user_model
                self.model_classes = model_classes
                self.is_binary = self.model_classes and len(self.model_classes) == 2

            def predict(self, data: pd.DataFrame) -> np.ndarray:
                """Convert labels to 0/1 if model is a binary classifier."""
                predicitions: np.ndarray = np.asarray(self.user_model.predict(data))
                # In case of binary converts into 0 and 1 the labels
                if self.is_binary:
                    transfer_func = np.vectorize(lambda x: 0 if x == self.model_classes[0] else 1)
                    predicitions = transfer_func(predicitions)
                # In case of multiclass with single label, convert into multi-label
                elif self.model_classes:
                    predicitions = _transform_to_multi_label_format(predicitions, self.model_classes)
                return predicitions

            def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
                """Validate model have predict_proba, and the proba matches the model classes."""
                if not hasattr(self.user_model, 'predict_proba'):
                    if isinstance(self.user_model, ClassifierMixin):
                        raise errors.DeepchecksValueError('Model is a sklearn classification model but lacks the'
                                                          ' predict_proba method. Please train the model with'
                                                          ' probability=True.')
                    raise errors.DeepchecksValueError(f'Scorer requires predicted probabilities, either implement '
                                                      f'predict_proba functionalities within the model objects or pass '
                                                      f'pre calculated probabilities. {SUPPORTED_MODELS_DOCLINK}')
                probabilities_per_class = self.user_model.predict_proba(data)
                validate_proba(probabilities_per_class, self.model_classes)
                return probabilities_per_class

            @property
            def classes_(self):
                return np.asarray([0, 1] if len(self.model_classes) == 2 else self.model_classes)

        return MyModelWrapper(model, self.model_classes)

    def _run_score(self, model, data: pd.DataFrame, label_col: pd.Series):
        # If scorer 'needs_threshold' or 'needs_proba' than the model has to have a predict_proba method.
        if ('needs' in self.scorer._factory_args()) and not hasattr(model,  # pylint: disable=protected-access
                                                                    'predict_proba'):
            raise errors.DeepchecksValueError(
                f'Can\'t compute scorer {self.scorer} when predicted probabilities are '
                f'not provided. Please use a model with predict_proba method or '
                f'manually provide predicted probabilities to the check. '
                f'{SUPPORTED_MODELS_DOCLINK}')
        if self.model_classes is not None:
            model = self._wrap_classification_model(model)
            if model.is_binary:
                if len(label_col.unique()) > 2:
                    raise errors.DeepchecksValueError('Model is binary but the label column has more than 2 classes: '
                                                      f'{label_col.unique()}')
                label_col = label_col.map({self.model_classes[0]: 0, self.model_classes[1]: 1})
            else:
                label_col = _transform_to_multi_label_format(np.array(label_col), self.model_classes)

        try:
            scores = self.scorer(model, data, np.array(label_col))
        except ValueError as e:
            if getattr(self.scorer, '_score_func', '').__name__ == 'roc_auc_score':
                get_logger().warning('ROC AUC failed with error message - "%s". setting scores as None', e,
                                     exc_info=get_logger().level == logging.DEBUG)
                scores = None
            else:
                raise

        # The scores returned are for the model classes but we want scores of the observed classes
        if self.model_classes is not None and isinstance(scores, t.Sized):
            if len(scores) != len(self.model_classes):
                raise errors.DeepchecksValueError(
                    f'Scorer returned {len(scores)} scores, but model contains '
                    f'{len(self.model_classes)} classes. Can\'t proceed')
            scores = dict(zip(self.model_classes, scores))
            # Add classes which been seen in the data but are not known to the model
            scores.update({cls: np.nan for cls in set(self.observed_classes) - set(self.model_classes)})

        return scores

    def __call__(self, model, dataset: 'tabular.Dataset'):
        """Run score with labels null filtering."""
        dataset_without_nulls = self.filter_nulls(dataset)
        return self._run_score(model, dataset_without_nulls.features_columns, dataset_without_nulls.label_col)

    def score_perfect(self, dataset: 'tabular.Dataset'):
        """Calculate the perfect score of the current scorer for given dataset."""
        dataset = self.filter_nulls(dataset)
        perfect_model = PerfectModel()
        perfect_model.fit(None, dataset.label_col)
        score = self._run_score(perfect_model, dataset.features_columns, dataset.label_col)
        if isinstance(score, dict):
            # We expect the perfect score to be equal for all the classes, so takes the first one
            score_values = np.asarray(list(score.values()))
            first_score = score_values[0]
            if any(score_values != first_score):
                get_logger().warning('Scorer %s return different perfect score for different classes', self.name)
            return first_score
        return score

    def validate_fitting(self, model, dataset: 'tabular.Dataset'):
        """Validate given scorer for the model and dataset."""
        dataset.assert_features()
        # In order for scorer to return result in right dimensions need to pass it samples from all labels
        single_label_data = dataset.data[dataset.label_col.notna()].groupby(dataset.label_name).head(1)
        new_dataset = dataset.copy(single_label_data)
        result = self._run_score(model, new_dataset.features_columns, new_dataset.label_col)

        if isinstance(result, dict):
            # Validate returns value for each class
            all_classes = set(self.model_classes) | set(self.observed_classes)
            if len(result) != len(all_classes):
                raise errors.DeepchecksValueError(f'Expected {len(all_classes)} classes, but scorer {self.name} '
                                                  f'returned {len(result)} elements in the score array value')
        elif result is not None and not isinstance(result, Number):
            raise errors.DeepchecksValueError(f'Expected scorer {self.name} to return number or dict '
                                              f'but got: {type(result).__name__}')


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
                          dataset: 'tabular.Dataset',
                          model_classes: t.Optional[t.List],
                          observed_classes: t.Optional[t.List]) -> t.List[DeepcheckScorer]:
    """Initialize scorers and return all of them as deepchecks scorers.

    Parameters
    ----------
    scorers : Mapping[str, Union[str, Callable]]
        dict of scorers names to scorer sklearn_name/function or a list without a name
    model : BasicModel
        used to validate the scorers, and calculate mode_type if None.
    dataset : Dataset
        used to validate the scorers, and calculate mode_type if None.
    model_classes: t.Optional[t.List]
        possible classes output for model. None for regression tasks.
    observed_classes: t.Optional[t.List]
        observed classes from labels and predictions. None for regression tasks.
    Returns
    --------
    scorers: t.List[DeepcheckScorer]
        A list of initialized DeepcheckScorers.
    """
    if isinstance(scorers, t.Mapping):
        scorers = [DeepcheckScorer(scorer, model_classes, observed_classes, name) for name, scorer in scorers.items()]
    else:
        scorers = [DeepcheckScorer(scorer, model_classes, observed_classes) for scorer in scorers]
    scorers: t.List[DeepcheckScorer]
    for s in scorers:
        s.validate_fitting(model, dataset)
    return scorers


def _transform_to_multi_label_format(y: np.ndarray, classes):
    # Some classifiers like catboost might return shape like (n_rows, 1), therefore squeezing the array.
    y = np.squeeze(y) if y.ndim > 1 else y
    if y.ndim == 1:
        binarizer = LabelBinarizer()
        binarizer.fit(classes)
        return binarizer.transform(y)
    # If after squeeze there are still 2 dimensions, then it must have column for each model class.
    elif y.ndim == 2 and y.shape[1] == len(classes):
        return y
    else:
        raise errors.DeepchecksValueError(f'got y with unworkable shape: {y.shape}. {SUPPORTED_MODELS_DOCLINK}')


def validate_proba(probabilities: np.array, model_classes: t.List):
    """Validate that the number of classes (columns) in probabilities matches the model_classes."""
    if probabilities.shape[1] != len(model_classes):
        raise errors.ModelValidationError(
            f'Model probabilities per class has {probabilities.shape[1]} '
            f'classes while known model classes has {len(model_classes)}. You can set the model\'s'
            f'classes manually using the model_classes argument in the run function.')
