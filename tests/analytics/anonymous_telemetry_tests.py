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
from deepchecks.analytics.anonymous_telemetry import get_environment_details
from hamcrest import assert_that, equal_to, is_in, has_key


def test_get_environment_details():
    env = get_environment_details()
    assert_that(env, has_key('python_version'))
    assert_that(env, has_key('os'))
    assert_that(env, has_key('deepchecks_version'))
    assert_that(env, has_key('runtime'))

    assert_that(env['runtime'], is_in(['docker', 'colab', 'notebook', 'paperspace', 'native']))
