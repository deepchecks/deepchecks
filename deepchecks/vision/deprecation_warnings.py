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

# Added in 0.7 Warning filters for deprecated functions in deepchecks.tabular.checks
# Should be removed in 0.8
warnings.filterwarnings(
    action='once',
    message=r'deepchecks.vision.checks.performance is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'deepchecks.vision.checks.methodology is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.tabular.checks.methodology.*'
)

warnings.filterwarnings(
    action='once',
    message=r'deepchecks.vision.checks.distribution is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='always',
    message=r'the integrity_validation suite is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

# Added in 0.7 Warning filters for drift conditions
# Should be removed in 0.8

warnings.filterwarnings(
    action='once',
    message=r'.*max_allowed_psi_score is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.*max_allowed_earth_movers_score is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='always',
    message=r'.*Property Warnings:.*',
    category=DeprecationWarning,
    module=r'deepchecks.vision.utils.*'
)
