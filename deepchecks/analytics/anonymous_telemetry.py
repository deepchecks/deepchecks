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
"""
Module for anonymous telemetry.

No credentials, data, personal information or anything private is collected (and will never be).
"""
import http.client
import os
import pathlib
import uuid
import warnings

import deepchecks
from deepchecks.utils.logger import get_logger

MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent
ANALYTICS_DISABLED = os.environ.get('DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY', False)


def validate_latest_version():
    """Check if we are on the latest version and send an anonymous import event to PostHog."""
    if not ANALYTICS_DISABLED:
        try:
            if os.path.exists(os.path.join(MODULE_DIR, '.user_id')):
                with open(os.path.join(MODULE_DIR, '.user_id'), 'r', encoding='utf8') as f:
                    user_id = f.read()
            else:
                user_id = str(uuid.uuid4())
                with open(os.path.join(MODULE_DIR, '.user_id'), 'w', encoding='utf8') as f:
                    f.write(user_id)

            conn = http.client.HTTPConnection('localhost', timeout=3)
            conn.request('GET', f'/v3/latest?version={deepchecks.__version__}&uuid={user_id}')
            result = conn.getresponse()
            is_on_latest = result.read().decode() == 'True'
            if not is_on_latest:
                get_logger().warning(f'You are using deepchecks version {deepchecks.__version__}, however a '
                                     f'newer version is available.'
                                     f'Deepchecks is frequently updated with major improvements. You should consider '
                                     f'upgrading via the "python -m pip install --upgrade deepchecks" command.')
        except Exception:  # pylint: disable=broad-except
            pass
