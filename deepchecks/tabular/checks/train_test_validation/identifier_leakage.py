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
"""module contains the Identifier Leakage check - deprecated."""

import warnings

from deepchecks.tabular.checks.data_integrity.identifier_label_correlation import IdentifierLabelCorrelation


class IdentifierLeakage(IdentifierLabelCorrelation):
    """Deprecated. Check if identifiers (Index/Date) can be used to predict the label."""

    def __init__(self, ppscore_params=None, **kwargs):
        warnings.warn('the identifier_leakage check is deprecated. use the identifier_label_correlation check instead',
                      DeprecationWarning, stacklevel=2)
        IdentifierLabelCorrelation.__init__(self, ppscore_params, **kwargs)
