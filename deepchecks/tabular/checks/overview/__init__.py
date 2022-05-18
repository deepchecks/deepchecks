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
"""
Module contains check of overall overview of datasets and model.

.. deprecated:: 0.6.5
        :mod:`deepchecks.tabular.checks.overview is deprecated and will be removed in deepchecks 0.8 version.
        Use :mod:`deepchecks.tabular.checks.integrity` and :mod:`deepchecks.tabular.checks.model_evaluation` instead.
"""
import warnings

from ..integrity import ColumnsInfo
from ..model_evaluation import ModelInfo

__all__ = [
    'ModelInfo',
    'ColumnsInfo'
]

warnings.warn(
                'deepchecks.tabular.checks.overview is deprecated. Use deepchecks.tabular.checks.model_evaluation '
                'and deepchecks.tabular.checks.integrity instead.',
                DeprecationWarning
            )
