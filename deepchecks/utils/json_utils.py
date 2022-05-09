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
# pylint: disable=unused-import,import-outside-toplevel, protected-access
"""Module with common utilities routines for serialization subpackage."""
import typing as t

import jsonpickle

from deepchecks.core.check_result import BaseCheckResult
from deepchecks.core.suite import SuiteResult

__all__ = ['from_json']


def from_json(json_dict: t.Union[str, t.Dict]) -> t.Union[BaseCheckResult, SuiteResult]:
    """Convert a json object that was returned from one of our classes to_json.

    Parameters
    ----------
    json_data: Union[str, Dict]
        Json data

    Returns
    -------
    Union[BaseCheckResult, SuiteResult]
        A check output or a suite result object.
    """
    if isinstance(json_dict, str):
        json_dict = jsonpickle.loads(json_dict)
    json_type = json_dict['type']
    if 'Check' in json_type:
        return BaseCheckResult.from_json(json_dict)
    if json_type == 'SuiteResult':
        return SuiteResult.from_json(json_dict)
    raise ValueError('Excpected json object to be one of '
                     '[CheckFailure, CheckResult, SuiteResult] but recievied: ' + json_type)
