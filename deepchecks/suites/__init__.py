# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module contains all prebuilt suites."""
from .integrity_suite import (
    single_dataset_integrity_suite,
    comparative_integrity_suite,
    integrity_suite
)
from .methodology_suite import (
    index_leakage_suite,
    date_leakage_suite,
    data_leakage_suite,
    leakage_suite,
    overfit_suite,
    methodological_flaws_suite
)
from .performance_suite import (
    classification_suite,
    regression_suite,
    generic_performance_suite,
    performance_suite
)
from .overall_suite import *


__all__ = [
    'single_dataset_integrity_suite',
    'comparative_integrity_suite',
    'integrity_suite',
    'index_leakage_suite',
    'date_leakage_suite',
    'data_leakage_suite',
    'leakage_suite',
    'overfit_suite',
    'methodological_flaws_suite',
    'classification_suite',
    'regression_suite',
    'generic_performance_suite',
    'regression_suite',
    'performance_suite'
]
