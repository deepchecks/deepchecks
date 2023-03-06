# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Public utils to be used by deepchecks users."""

import typing as t

import pandas as pd

from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular import Dataset
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.tabular.utils.feature_importance import _calculate_feature_importance

__all__ = ['calculate_feature_importance']

from deepchecks.tabular.utils.task_inference import (get_all_labels, infer_classes_from_model,
                                                     infer_task_type_by_class_number, infer_task_type_by_labels)


def calculate_feature_importance(
        model: t.Any,
        dataset: Dataset,
        n_repeats: int = 30,
        mask_high_variance_features: bool = False,
        n_samples: int = 10_000,
        alternative_scorer: t.Optional[DeepcheckScorer] = None,
        force_permutation: bool = False,
        random_state: int = 42
) -> pd.Series:
    """
    Get or calculate feature importance outside of check and suite.

    Many checks and suites in deepchecks use feature importance as part of its calculation or output. If your model
    does not have built-in feature importance, the check or suite will calculate it for you. This calculation is done
    in every call to the check or suite ``run`` function. Therefore, when running different checks outside a suite,
    or running the same suite several times, this calculation will be done every time.

    The recalculation can be avoided by calculating the feature importance in advance. Use this function to calculate
    your feature importance, and then give it as an input to the check or suite ``run`` function, as follows:

    >>> from deepchecks.tabular.feature_importance import calculate_feature_importance
    >>> from deepchecks.tabular.datasets.classification.iris import load_data, load_fitted_model
    >>> from deepchecks.tabular.checks import UnusedFeatures
    >>> _, iris_test_dataset = load_data()
    >>> iris_model = load_fitted_model()
    >>> fi = calculate_feature_importance(model=iris_model, dataset=iris_test_dataset)
    >>> result = UnusedFeatures().run(iris_test_dataset, model=iris_model, feature_importance=fi)

    By defualt this function will attempt to get the feature importance from the model. If the model does not have
    built-in feature importance, it will calculate it using permutation importance. If you want to force the
    calculation of the feature importance, even if the model has built-in feature importance, use the
    ``force_permutation`` parameter.

    Parameters
    ----------
    model: t.Any
        a fitted model
    dataset: t.Union['tabular.Dataset', pd.DataFrame]
        dataset used to fit the model
    n_repeats: int, default: 30
        Number of times to permute a feature
    mask_high_variance_features : bool , default: False
        If true, features for which calculated permutation importance values
        varied greatly would be returned has having 0 feature importance
    n_samples: int, default: 10_000
        The number of samples to draw from X to compute feature importance
        in each repeat (without replacement).
    alternative_scorer: t.Optional[DeepcheckScorer], default: None
        Scorer to use for evaluation of the model performance in the permutation_importance function. If not defined,
        the default deepchecks scorers are used.
    force_permutation : bool, default: False
        Force permutation importance calculation.
    random_state: int, default: 42
        Random seed for permutation importance calculation.

    Returns
    -------
    pd.Series:
        feature importance normalized to 0-1 indexed by feature names
    """
    permutation_kwargs = {
        'n_repeats': n_repeats,
        'mask_high_variance_features': mask_high_variance_features,
        'n_samples': n_samples,
        'alternative_scorer': alternative_scorer,
        'random_state': random_state,
        'skip_messages': False,
        'timeout': None
    }

    if isinstance(dataset, pd.DataFrame):
        raise DeepchecksValueError('Cannot calculate permutation feature importance on a pandas Dataframe. '
                                   'In order to force permutation feature importance, please use the Dataset'
                                   ' object.')

    model_classes = infer_classes_from_model(model)
    labels = get_all_labels(model, dataset)
    observed_classes = sorted(labels.dropna().unique().tolist())

    if dataset.label_type:
        task_type = dataset.label_type
    elif model_classes:
        task_type = infer_task_type_by_class_number(len(model_classes))
    else:
        task_type = infer_task_type_by_labels(labels)

    fi, _ = _calculate_feature_importance(model=model,
                                          dataset=dataset,
                                          model_classes=model_classes or observed_classes,
                                          observed_classes=observed_classes,
                                          task_type=task_type,
                                          force_permutation=force_permutation,
                                          permutation_kwargs=permutation_kwargs)
    return fi
