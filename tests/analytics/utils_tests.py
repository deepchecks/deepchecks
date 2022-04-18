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
import os.path

import yaml

from deepchecks.analytics.utils import write_global_telemetry_config, CONFIG_FILE_PATH, get_telemetry_config, \
    toggle_telemetry, is_telemetry_enabled


def test_write_global_telemetry():
    try:
        os.remove(os.path.expanduser(CONFIG_FILE_PATH))
    except FileNotFoundError:
        pass

    write_global_telemetry_config()

    with open(os.path.expanduser(CONFIG_FILE_PATH), 'r', encoding='utf8') as f:
        content = f.read()
        config = yaml.safe_load(content)

        assert config['telemetry_enabled'] is True
        assert 'machine_id' in config

    toggle_telemetry(False)


def test_get_telemetry_config():
    config = get_telemetry_config()

    assert config['telemetry_enabled'] is False
    assert 'machine_id' in config


def test_toggle_telemetry():
    toggle_telemetry(True)
    config = get_telemetry_config()
    assert config['telemetry_enabled'] is True

    toggle_telemetry(False)
    config = get_telemetry_config()
    assert config['telemetry_enabled'] is False


def test_is_telemetry_enabled():
    value = is_telemetry_enabled()
    with open(os.path.expanduser(CONFIG_FILE_PATH), 'r', encoding='utf8') as f:
        content = f.read()
        config = yaml.safe_load(content)

        assert value is config['telemetry_enabled']
