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
"""Utils module with methods for fast calculations."""

from deepchecks.core.errors import DeepchecksTimeoutError


def get_time_out_handler(error_message: str):
    """Get a timeout handler."""
    def timeout_handler(signum, frame):  # Custom signal handler
        raise DeepchecksTimeoutError(error_message)

    return timeout_handler
