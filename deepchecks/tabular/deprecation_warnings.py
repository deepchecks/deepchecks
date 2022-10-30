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

warnings.filterwarnings(
    action='once',
    message=r'The SegmentPerformance check is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'The WholeDatasetDrift check is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* label type is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* alternative_scorers is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* y_pred_train is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* y_proba_train is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* y_pred_test is deprecated and ignored.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)

warnings.filterwarnings(
    action='once',
    message=r'.* y_proba_test is deprecated and ignored.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)


warnings.filterwarnings(
    action='once',
    message=r'.*RegressionSystematicError check is deprecated.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)
