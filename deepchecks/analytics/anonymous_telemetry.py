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
import json
import os
import pathlib
import platform

import deepchecks
from deepchecks import is_notebook
from deepchecks.analytics.utils import get_telemetry_config

MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent
ANALYTICS_ENABLED = get_telemetry_config()['telemetry_enabled']


def _identify_runtime():
    """Identify the runtime."""
    # If in docker
    try:
        with open('/proc/1/cgroup', 'rt', encoding='utf8') as f:
            info = f.read()
        if 'docker' in info or 'kubepods' in info:
            return 'docker'
    except (FileNotFoundError, Exception):  # pylint: disable=broad-except
        pass

    # If in colab
    if 'COLAB_GPU' in os.environ:
        return 'colab'

    # If in notebook
    if is_notebook():
        return 'notebook'

    # If in paperspace
    if 'PAPERSPACE_NOTEBOOK_REPO_ID' in os.environ:
        return 'paperspace'

    return 'native'


try:
    RUNTIME = _identify_runtime()
except Exception:  # pylint: disable=broad-except
    RUNTIME = 'unknown'


def get_environment_details():
    """Get environment details."""
    return {
        'python_version': platform.python_version(),
        'os': platform.system(),
        'deepchecks_version': deepchecks.__version__,
        'runtime': RUNTIME,
    }


def send_anonymous_event(event_name, event_data=None):
    """Send an anonymous event ."""
    if ANALYTICS_ENABLED:
        try:
            machine_id = get_telemetry_config()['machine_id']
            params = {
                'event_name': event_name,
                'event_data': event_data,
                'machine_id': machine_id,
            }
            params.update(get_environment_details())

            conn = http.client.HTTPConnection('localhost:80', timeout=3)
            conn.request('POST', '/events', body=json.dumps(params))
            _ = conn.getresponse()
        except Exception:  # pylint: disable=broad-except
            pass


def send_anonymous_run_event(run_instance):
    """Send an anonymous check run event."""
    from deepchecks import BaseSuite, BaseCheck  # pylint: disable=import-outside-toplevel
    if isinstance(run_instance, BaseCheck):
        event_name = 'run-check'
    elif isinstance(run_instance, BaseSuite):
        event_name = 'run-suite'
    else:
        event_name = 'run-unknown'

    try:
        mod = run_instance.__module__

        def get_custom_check_kind(clazz):
            """Get first deepchecks ancestor."""
            mro = clazz.__class__.__mro__
            for cls in mro:
                if cls.__module__.startswith('deepchecks'):
                    return cls.p__module__.split('.')[1]
            return 'unknown'

        if mod.startswith('deepchecks'):
            try:
                name = run_instance.name()
            except Exception:  # pylint: disable=broad-except
                name = run_instance.__class__.__name__
            event_data = {
                'name': name,
                'kind': mod.split('.')[1],
            }
        else:
            event_data = {
                'name': 'custom_check',
                'kind': get_custom_check_kind(run_instance)
            }

    except Exception:  # pylint: disable=broad-except
        event_data = None

    send_anonymous_event(event_name, event_data)
