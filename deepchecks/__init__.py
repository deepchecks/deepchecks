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
"""Deepchecks."""
import os
import pathlib
import http.client
import warnings
import matplotlib
import plotly.io as pio
from pkg_resources import parse_version

from deepchecks.utils.ipython import is_notebook
from deepchecks.tabular import (
    Dataset,
    Suite,
    Context,
    SingleDatasetCheck,
    TrainTestCheck,
    ModelOnlyCheck,
    ModelComparisonCheck,
    ModelComparisonSuite,
)
from deepchecks.core import (
    BaseCheck,
    BaseSuite,
    CheckResult,
    CheckFailure,
    SuiteResult,
    Condition,
    ConditionResult,
    ConditionCategory,
    SingleDatasetBaseCheck,
    TrainTestBaseCheck,
    ModelOnlyBaseCheck
)


warnings.warn(
    # TODO: better message
    'Ability to import base tabular functionality from '
    'the `deepchecks` directly is deprecated, please import from '
    '`deepchecks.tabular` instead',
    DeprecationWarning
)


__all__ = [
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'ModelComparisonCheck',
    'CheckResult',
    'CheckFailure',
    'Condition',
    'ConditionResult',
    'ConditionCategory',
    'BaseSuite',
    'SuiteResult',

    # tabular checks
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'Dataset',
    'Suite',
    'ModelComparisonSuite',
    'Context'
]


# Matplotlib has multiple backends. If we are in a context that does not support GUI (For example, during unit tests)
# we can't use a GUI backend. Thus we must use a non-GUI backend.
if not is_notebook():
    matplotlib.use('Agg')

# We can't rely on that the user will have an active internet connection, thus we change the default backend to
# "notebook" If plotly detects the 'notebook-connected' backend.
# for more info, see: https://plotly.com/python/renderers/
pio_backends = pio.renderers.default.split('+')
if 'notebook_connected' in pio_backends:
    pio_backends[pio_backends.index('notebook_connected')] = 'notebook'
    pio.renderers.default = '+'.join(pio_backends)


# Set version info
try:
    MODULE_DIR = pathlib.Path(__file__).absolute().parent.parent
    with open(os.path.join(MODULE_DIR, 'VERSION'), 'r', encoding='utf-8') as f:
        __version__ = f.read().strip()
except:  # pylint: disable=bare-except # noqa
    # If version file can't be found, leave version empty
    __version__ = ''

# Check for latest version
try:
    disable = os.environ.get('DEEPCHECKS_DISABLE_LATEST', 'false').lower() == 'true'
    if not disable:
        conn = http.client.HTTPSConnection('api.deepchecks.com', timeout=3)
        conn.request('GET', '/latest')
        response = conn.getresponse()
        latest_version = response.read().decode('utf-8')
        if __version__ and parse_version(__version__) < parse_version(latest_version):
            warnings.warn('Looks like you are using outdated version of deepchecks. consider upgrading using'
                          ' pip install -U deepchecks')
except:  # pylint: disable=bare-except # noqa
    pass
