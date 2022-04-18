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
"""Utils for analytics."""
import os
import uuid
from typing import Dict, Text, Any

import yaml

CONFIG_FILE_PATH = '~/.deepchecks/config.yml'


def _default_telemetry_config(is_enabled: bool) -> Dict[Text, Any]:
    """Return the default telemetry config.

    Parameters
    ----------
    is_enabled : bool
        Whether or not to enable telemetry.

    Returns
    -------
    Dict[Text, Any]
        The default telemetry config.
    """
    return {
        'telemetry_enabled': is_enabled,
        'machine_id': uuid.uuid4().hex,
    }


def write_global_telemetry_config():
    """Write the default telemetry config to the config file."""
    if not os.path.exists(os.path.dirname(os.path.expanduser(CONFIG_FILE_PATH))):
        os.makedirs(os.path.dirname(os.path.expanduser(CONFIG_FILE_PATH)))
    if not os.path.exists(os.path.expanduser(CONFIG_FILE_PATH)):
        with open(os.path.expanduser(CONFIG_FILE_PATH), 'w', encoding='utf8') as f:
            f.write(
                yaml.dump(
                    _default_telemetry_config(True),
                ))


def get_telemetry_config() -> Dict[Text, Any]:
    """Return the telemetry config from the config file.

    Returns
    ------
    Dict[Text, Any]
        The telemetry config.
    """
    if not os.path.exists(os.path.expanduser(CONFIG_FILE_PATH)):
        write_global_telemetry_config()
    with open(os.path.expanduser(CONFIG_FILE_PATH), 'r', encoding='utf8') as f:
        return yaml.safe_load(f)


def toggle_telemetry(value: bool):
    """Disable telemetry."""
    config = get_telemetry_config()
    config['telemetry_enabled'] = value
    with open(os.path.expanduser(CONFIG_FILE_PATH), 'w', encoding='utf8') as f:
        f.write(
            yaml.dump(
                config,
            ))


def is_telemetry_enabled() -> bool:
    """Return whether or not telemetry is enabled."""
    return get_telemetry_config()['telemetry_enabled']
