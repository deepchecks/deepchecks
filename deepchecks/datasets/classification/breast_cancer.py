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
"""Alternative way to import tabular breast_cancer dataset.

This module exists only for backward compatibility and will be
removed in the nexts versions.
"""
from deepchecks.tabular.datasets.classification.breast_cancer import (
    load_data, load_fitted_model)

__all__ = ['load_data', 'load_fitted_model']
