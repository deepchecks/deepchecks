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
"""Module containing the check results classes."""
# pylint: disable=super-init-not-called
import base64
import io
from typing import Any, Dict, List, Union

import jsonpickle
import pandas as pd
import plotly

from deepchecks.core.check_result import CheckFailure, CheckResult, DisplayMap
from deepchecks.core.condition import Condition, ConditionCategory, ConditionResult
from deepchecks.utils.html import imagetag

__all__ = [
    'CheckResultJson',
    'CheckFailureJson',
]


class FakeCheck:
    def __init__(self, metadata: Dict):
        self._metadata = metadata

    def metadata(self, *args, **kwargs):  # pylint: disable=unused-argument
        return self._metadata

    def name(self):
        return self._metadata['name']


class CheckResultJson(CheckResult):
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Parameters
    ----------
    json_data: Union[str, Dict]
        Json data
    """

    def __init__(self, json_dict: Union[str, Dict]):
        if isinstance(json_dict, str):
            json_dict = jsonpickle.loads(json_dict)

        self.value = json_dict.get('value')
        self.header = json_dict.get('header')
        self.check = FakeCheck(json_dict.get('check'))

        conditions_results_json = json_dict.get('conditions_results')

        if conditions_results_json is not None:
            self.conditions_results = []
            for condition in conditions_results_json:
                cond_res = ConditionResult(ConditionCategory[condition['Status']], condition['More Info'])
                cond_res.set_name(condition['Condition'])
                self.conditions_results.append(cond_res)
        else:
            self.conditions_results = None

        json_display = json_dict.get('display', [])
        self.display = self._process_jsonified_display_items(json_display)

    def process_conditions(self) -> List[Condition]:
        """Conditions are already processed it is to prevent errors."""
        pass

    @classmethod
    def _process_jsonified_display_items(cls, display: List[Dict[str, Any]]) -> List[Any]:
        assert isinstance(display, list)
        output = []

        for record in display:
            display_type, payload = record['type'], record['payload']
            if display_type == 'html':
                output.append(payload)
            elif display_type == 'dataframe':
                df = pd.DataFrame.from_records(payload)
                output.append(df)
            elif display_type == 'plotly':
                plotly_json = io.StringIO(payload)
                output.append(plotly.io.read_json(plotly_json))
            elif display_type == 'plt':
                output.append((f'<img src=\'data:image/png;base64,{payload}\'>'))
            elif display_type == 'images':
                assert isinstance(payload, list)
                output.extend(imagetag(base64.b64decode(it)) for it in payload)
            elif display_type == 'displaymap':
                assert isinstance(payload, dict)
                output.append(DisplayMap(**{
                    k: cls._process_jsonified_display_items(v)
                    for k, v in payload.items()
                }))
            else:
                raise ValueError(f'Unexpected type of display received: {display_type}')

        return output


class CheckFailureJson(CheckFailure):
    """Class which holds a check run exception.

    Parameters
    ----------
    json_data: Union[str, Dict]
        Json data
    """

    def __init__(self, json_dict: Union[str, Dict]):
        if isinstance(json_dict, str):
            json_dict = jsonpickle.loads(json_dict)

        self.header = json_dict.get('header')
        self.check = FakeCheck(json_dict.get('check'))
        self.exception = json_dict.get('exception')

    def print_traceback(self):
        """Print the traceback of the failure."""
        print(self.exception)
