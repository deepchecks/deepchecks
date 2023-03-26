import abc
import typing as t
import warnings
import numpy as np
import pandas as pd
from deepchecks.core.checks import DatasetKind

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.recommender.dataset import RecDataset
from deepchecks.recommender.context import Context as RecContext

from deepchecks.core.checks import DatasetKind
from deepchecks.recommender import Context

def single_run(
    self,
    dataset: 'RecDataset',
    model: t.Optional['BasicModel'] = None,
    item_dataset: t.Optional['ItemDataset'] = None,
    feature_importance: t.Optional[pd.Series] = None,
    feature_importance_force_permutation: bool = False,
    feature_importance_timeout: int = 120,
    with_display: bool = True,
    y_pred_train: t.Optional[np.ndarray] = None,
    y_pred_test: t.Optional[np.ndarray] = None,
    model_classes: t.Optional[t.List] = None
) -> 'CheckResult':
    """Run check.

    Parameters
    ----------
    dataset: Union[Dataset, pd.DataFrame]
        Dataset or DataFrame object, representing data an estimator was fitted on
    model: Optional[BasicModel], default: None
        A scikit-learn-compatible fitted estimator instance
    {additional_context_params:2*indent}
    """
    if self.context_type is None:
        self.context_type = Context

    context = self.context_type(  # pylint: disable=not-callable
        train=dataset,
        model=model,
        item_dataset=item_dataset,
        feature_importance=feature_importance,
        feature_importance_force_permutation=feature_importance_force_permutation,
        feature_importance_timeout=feature_importance_timeout,
        with_display=with_display,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        model_classes=model_classes
    )
    result = self.run_logic(context, dataset_kind=DatasetKind.TRAIN)
    context.finalize_check_result(result, self, DatasetKind.TRAIN)
    return result

def train_test_run(
    self,
    dataset: 'RecDataset',
    model: t.Optional['BasicModel'] = None,
    item_dataset: t.Optional['ItemDataset'] = None,
    feature_importance: t.Optional[pd.Series] = None,
    feature_importance_force_permutation: bool = False,
    feature_importance_timeout: int = 120,
    with_display: bool = True,
    y_pred_train: t.Optional[np.ndarray] = None,
    y_pred_test: t.Optional[np.ndarray] = None,
    model_classes: t.Optional[t.List] = None
) -> 'CheckResult':
    """Run check.

    Parameters
    ----------
    dataset: Union[Dataset, pd.DataFrame]
        Dataset or DataFrame object, representing data an estimator was fitted on
    model: Optional[BasicModel], default: None
        A scikit-learn-compatible fitted estimator instance
    {additional_context_params:2*indent}
    """
    if self.context_type is None:
        self.context_type = Context

    context = self.context_type(  # pylint: disable=not-callable
        train=dataset,
        model=model,
        item_dataset=item_dataset,
        feature_importance=feature_importance,
        feature_importance_force_permutation=feature_importance_force_permutation,
        feature_importance_timeout=feature_importance_timeout,
        with_display=with_display,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        model_classes=model_classes
    )
    result = self.run_logic(context, dataset_kind=DatasetKind.TRAIN)
    context.finalize_check_result(result, self, DatasetKind.TRAIN)
    return result
