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

import deepchecks

MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent
ANALYTICS_DISABLED = os.environ.get('DISABLE_DEEPCHECKS_ANONYMOUS_TELEMETRY', False)


def send_anonymous_import_event():
    """Send an anonymous import event to PostHog."""
    if not ANALYTICS_DISABLED:
        try:
            if os.path.exists(os.path.join(MODULE_DIR, '.user_id')):
                with open(os.path.join(MODULE_DIR, '.user_id'), 'r', encoding='utf8') as f:
                    user_id = f.read()
            else:
                user_id = str(uuid.uuid4())
                with open(os.path.join(MODULE_DIR, '.user_id'), 'w', encoding='utf8') as f:
                    f.write(user_id)

            conn = http.client.HTTPSConnection('api.deepchecks.com', timeout=3)
            conn.request('GET', f'/metrics?version={deepchecks.__version__}&uuid={user_id}')
            _ = conn.getresponse()
        except Exception:  # pylint: disable=broad-except
            pass
