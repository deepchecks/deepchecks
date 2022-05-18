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
"""Module containing all data distribution checks.

.. deprecated:: 0.7.0
        :mod:`deepchecks.tabular.checks.distribution is deprecated and will be removed in deepchecks 0.8 version.
        Use :mod:`deepchecks.tabular.checks.train_test_validation` instead.
"""
import warnings

from ..model_evaluation import TrainTestPredictionDrift
from ..train_test_validation import (TrainTestFeatureDrift,
                                     TrainTestLabelDrift, WholeDatasetDrift)

__all__ = [
    'TrainTestFeatureDrift',
    'WholeDatasetDrift',
    'TrainTestLabelDrift',
    'TrainTestPredictionDrift'
]

warnings.filterwarnings(
    action='once',
    message=r'deepchecks.tabular.checks.distribution is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.tabular.checks.distribution.*'
)

warnings.warn(
                'deepchecks.tabular.checks.distribution is deprecated. Use '
                'deepchecks.tabular.checks.train_test_validation instead.',
                DeprecationWarning
            )
