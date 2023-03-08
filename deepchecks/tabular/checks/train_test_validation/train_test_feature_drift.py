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

"""This module contains the code that handles users' use of the deprecated TrainTestFeatureDrift.

Its name was changed to FeatureDrift (removed the TrainTest prefix)
"""

import warnings

from deepchecks.tabular.checks.train_test_validation.feature_drift import FeatureDrift


class TrainTestFeatureDrift(FeatureDrift):
    """The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version.

    .. deprecated:: 0.14.0
        `deepchecks.tabular.checks.TrainTestFeatureDrift is deprecated and will be removed in deepchecks 0.14 version.
        Use `deepchecks.tabular.checks.FeatureDrift` instead.

    Please use the FeatureDrift check instead
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestFeatureDrift check is deprecated and will be removed in the 0.14 version."
                      " Please use the FeatureDrift check instead", DeprecationWarning, stacklevel=2)
        FeatureDrift.__init__(self, *args, **kwargs)
