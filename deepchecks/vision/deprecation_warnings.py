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
"""This file changes default 'ignore' action of DeprecationWarnings for specific deprecation messages."""
import warnings

warnings.filterwarnings(
    action='once',
    message=r'.*check is deprecated and will be removed in the 0.14 version.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)
