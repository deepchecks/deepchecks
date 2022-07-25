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
from deepchecks.utils.features import calculate_feature_importance as _calculate_feature_importance

__all__ = [
    'calculate_feature_importance'
]


def calculate_feature_importance(
        model: t.Any,
        dataset: t.Union['tabular.Dataset', pd.DataFrame],
        permutation_kwargs: t.Dict[str, t.Any] = None,
) -> pd.Series:
    permutation_kwargs = permutation_kwargs or {}
    permutation_kwargs['skip_messages'] = permutation_kwargs.get('skip_messages', True)
    permutation_kwargs['timeout'] = permutation_kwargs.get('timeout', None)

    if isinstance(dataset, pd.DataFrame):
        raise DeepchecksValueError('Cannot calculate permutation feature importance on a pandas Dataframe. '
                                   'In order to force permutation feature importance, please use the Dataset'
                                   ' object.')

    fi, _ = _calculate_feature_importance(model=model, dataset=dataset, force_permutation=True,
                                          permutation_kwargs=permutation_kwargs)
    return fi
