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

"""This module contains the code that handles users' use of the deprecated TrainTestLabelDrift.

Its name was changed to LabelDrift (removed the TrainTest prefix)
"""
import warnings

from deepchecks.tabular.checks.train_test_validation.label_drift import LabelDrift


class TrainTestLabelDrift(LabelDrift):
    """The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version.

    Please use the LabelDrift check instead.
    """

    def __init__(self, *args, **kwargs):
        warnings.warn("The TrainTestLabelDrift check is deprecated and will be removed in the 0.14 version."
                      "Please use the LabelDrift check instead.", DeprecationWarning, stacklevel=2)
        LabelDrift.__init__(self, *args, **kwargs)
