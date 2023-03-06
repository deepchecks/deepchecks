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
"""This module contains the code that handles users' use of the deprecated TrainTestSamplesMix, which names
 was changed to SamplesMix (removed the TrainTest prefix)"""
import warnings
from deepchecks.tabular.checks.train_test_validation.samples_mix import SamplesMix
from deepchecks.utils.deprecation import DeprecationHelper

_deprecation_message = 'The TrainTestSamplesMix check is deprecated and will be removed in the 0.14 version.' \
                       ' Please use the SamplesMix check instead.'
warnings.warn(_deprecation_message)

TrainTestSamplesMix = DeprecationHelper(SamplesMix, _deprecation_message)
