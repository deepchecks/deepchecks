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
"""Module contains all prebuilt vision suites."""
from .default_suites import data_integrity, full_suite, integrity_validation, model_evaluation, train_test_validation

__all__ = ['train_test_validation',
           'model_evaluation',
           'full_suite',
           'integrity_validation',
           'data_integrity']
