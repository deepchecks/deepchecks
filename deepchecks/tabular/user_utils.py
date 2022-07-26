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
"""Public utils to be used by deepchecks users."""

import typing as t
import pandas as pd

from deepchecks import tabular
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.tabular.metric_utils import DeepcheckScorer
from deepchecks.utils.features import calculate_feature_importance as _calculate_feature_importance

__all__ = [
    'feature_importance'
]


def feature_importance(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
        n_repeats: int = 30,
        mask_high_variance_features: bool = False,
        n_samples: int = 10_000,
        alternative_scorer: t.Optional[DeepcheckScorer] = None,
        random_state: int = 42
) -> pd.Series:
    """
    Calculate feature importance outside of check and suite.

    As many checks and suites use feature importance, it is rec

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

    fi, _ = _calculate_feature_importance(model=model, dataset=dataset, force_permutation=True,
                                          permutation_kwargs=permutation_kwargs)
    return fi
