# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""The category mismatch train-test check module."""
import warnings

from .new_category_train_test import NewCategoryTrainTest


class CategoryMismatchTrainTest(NewCategoryTrainTest):
    """Find new categories in the test set."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        warnings.warn('CategoryMismatchTrainTest is deprecated, use NewCategoryTrainTest instead',
                      DeprecationWarning)

    pass
