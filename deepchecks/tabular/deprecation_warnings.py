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
"""This file changes default 'ignore' action of DeprecationWarnings for specific deprecation messages."""
import warnings

# Added in version 0.6.2, deprecates max_num_categories in all drift checks
warnings.filterwarnings(
    action='always',
    message=r'.*max_num_categories.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)
