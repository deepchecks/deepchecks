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
import sys
import types
import warnings
from importlib._bootstrap import _init_module_attrs

try:
    from importlib.metadata import version
except ImportError:
    # NOTE:
    # 'importlib.metadata' was added to python>=3.8
    # for versions that are <3.8 we need to install
    # 'importlib_metadata' package
    from importlib_metadata import version

# NOTE: it is here, before other import, in order to omit circular import error
__version__ = version('deepchecks')

import matplotlib
import plotly.io as pio

from deepchecks.analytics.anonymous_telemetry import validate_latest_version
from deepchecks.core import (BaseCheck, BaseSuite, CheckFailure, CheckResult, Condition, ConditionCategory,
                             ConditionResult, ModelOnlyBaseCheck, SingleDatasetBaseCheck, SuiteResult,
                             TrainTestBaseCheck)
# TODO: remove in further versions
from deepchecks.tabular import (Context, Dataset, ModelComparisonCheck, ModelComparisonSuite, ModelOnlyCheck,
                                SingleDatasetCheck, Suite, TrainTestCheck)
from deepchecks.utils.ipython import is_notebook
from deepchecks.utils.logger import get_verbosity, set_verbosity

__all__ = [
    # core
    'BaseCheck',
    'SingleDatasetBaseCheck',
    'TrainTestBaseCheck',
    'ModelOnlyBaseCheck',
    'CheckResult',
    'CheckFailure',
    'Condition',
    'ConditionResult',
    'ConditionCategory',
    'BaseSuite',
    'SuiteResult',
    # tabular
    'Dataset',
    'Suite',
    'Context',
    'SingleDatasetCheck',
    'TrainTestCheck',
    'ModelOnlyCheck',
    'ModelComparisonCheck',
    'ModelComparisonSuite',
    # logger
    'set_verbosity',
    'get_verbosity',
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


# Send an import event if not disabled
validate_latest_version()

# ================================================================

warnings.filterwarnings(
    action='once',
    message=r'Ability to import.*',
    category=DeprecationWarning,
    module=r'deepchecks.*'
)


# NOTE:
# Code below is a temporary hack that exists only to provide backward compatibility
# and will be removed in the next versions.

# TODO:
# python (>=3.7) provides an ability to modify module attribute access by
# implementing '__getattr__' function on the module level scope

__original_module__ = sys.modules[__name__]


class _SubstituteModule(types.ModuleType):
    """Substitute module type to provide backward compatibility."""

    ROUTINES = (
        'Dataset',
        'Suite',
        'Context',
        'SingleDatasetCheck',
        'TrainTestCheck',
        'ModelOnlyCheck',
        'ModelComparisonCheck',
        'ModelComparisonSuite',
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__.update(__original_module__.__dict__)

    def __getattribute__(self, name):
        routines = object.__getattribute__(self, 'ROUTINES')
        if name in routines:
            deprecation_warning = 'Ability to import base tabular functionality from the `deepchecks` package ' \
                                  'directly is deprecated, please import from `deepchecks.tabular` instead'

            if os.environ.get('FAIL_ON_DEEPCHECKS_DEPRECATION_WARNINGS') == 'true':
                raise DeprecationWarning(deprecation_warning)
            warnings.warn(
                deprecation_warning,
                DeprecationWarning
            )
        return object.__getattribute__(self, name)


__substitute_module__ = _SubstituteModule(__name__)
_init_module_attrs(__original_module__.__spec__, __substitute_module__, override=True)
sys.modules[__name__] = __substitute_module__
