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
"""Module contains the WholeDatasetDrift check - deprecated."""

import warnings

from deepchecks.tabular.checks.train_test_validation import MultivariateDrift


class WholeDatasetDrift(MultivariateDrift):
    """
    Calculate drift between the entire train and test datasets using a model trained to distinguish between them.

    .. deprecated:: 0.9
        The WholeDatasetDrift check is deprecated and will be removed in the 0.11 version. Please use the
        MultivariateDrift check instead.
    """

    def __init__(
            self,
            n_top_columns: int = 3,
            min_feature_importance: float = 0.05,
            max_num_categories_for_display: int = 10,
            show_categories_by: str = 'largest_difference',
            n_samples: int = 10_000,
            random_state: int = 42,
            test_size: float = 0.3,
            min_meaningful_drift_score: float = 0.05,
            **kwargs
    ):

        warnings.warn(
            'The WholeDatasetDrift check is deprecated and will be removed in the 0.11 version. '
            'Please use the MultivariateDrift check instead.',
            DeprecationWarning, stacklevel=2
        )

        MultivariateDrift.__init__(
            self, n_top_columns,
            min_feature_importance,
            max_num_categories_for_display,
            show_categories_by,
            n_samples,
            random_state,
            test_size,
            min_meaningful_drift_score,
            **kwargs
        )
