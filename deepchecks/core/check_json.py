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
# pylint: disable=broad-except
import io

from typing import List

import jsonpickle
import jsonpickle.ext.pandas as jsonpickle_pd
import pandas as pd
import plotly
from deepchecks.core.check_result import CheckFailure, CheckResult

from deepchecks.core.condition import (Condition, ConditionCategory,
                                       ConditionResult)

# registers jsonpickle pandas extension for pandas support in the to_json function
jsonpickle_pd.register_handlers()

__all__ = [
    'CheckJson',
    'CheckJsonFailure',
]


class CheckJson(CheckResult):
    """Class which returns from a check with result that can later be used for automatic pipelines and display value.

    Class containing the result of a check

    The class stores the results and display of the check. Evaluating the result in an IPython console / notebook
    will show the result display output.

    Parameters
    ----------
    json_data : str
        Json data
    """

    def __init__(self, json_data: str):
        json_dict = jsonpickle.loads(json_data)

        self.value = json_dict.get('value')
        self._check_name = json_dict.get('name')
        self.header = json_dict.get('header')
        self.params = json_dict.get('params')
        self.summary = json_dict.get('summary')

        conditions_table = json_dict.get('conditions_table')
        if conditions_table is not None:
            self.conditions_results = []
            conditions_table = jsonpickle.loads(conditions_table)
            for condition in conditions_table:
                cond_res = ConditionResult(ConditionCategory[condition['Status']], condition['More Info'])
                cond_res.set_name(condition['Condition'])
                self.conditions_results.append(cond_res)
        else:
            self.conditions_results = None
        json_display = json_dict.get('display')
        self.display = []
        if json_display is not None:
            for display_type, value in json_display:
                if display_type == 'html':
                    self.display.append(value)
                elif display_type == 'dataframe':
                    df: pd.DataFrame = pd.read_json(value, orient='records')
                    self.display.append(df)
                elif display_type == 'plotly':
                    plotly_json = io.StringIO(value)
                    self.display.append(plotly.io.read_json(plotly_json))
                elif display_type == 'plt':
                    self.display.append((f'<img src=\'data:image/png;base64,{value}\'>'))
                else:
                    raise ValueError(f'Unexpected type of display received: {display_type}')

    def _get_metadata(self, _: bool = False):
        header = self.get_header()
        return {'name': self.check_name, 'params': self.params, 'header': header,
                'summary': self.summary}

    def get_header(self) -> str:
        """Return header for display. if header was defined return it, else extract name of check class."""
        return self.header

    def process_conditions(self) -> List[Condition]:
        """Conditions are already processed it is to prevent errors."""
        pass

    @property
    def check_name(self):
        return self._check_name

class CheckJsonFailure(CheckFailure):
    """Class which holds a check run exception.

    Parameters
    ----------
    check : BaseCheck
    exception : Exception
    header_suffix : str , default ``

    """

    def __init__(self, json_data: str):
        json_dict = jsonpickle.loads(json_data)
        self.exception = json_dict.get('exception')
        self._check_name = json_dict.get('name')
        self.header = json_dict.get('header')
        self.params = json_dict.get('params')
        self.summary = json_dict.get('summary')

    def _get_metadata(self, _: bool = False):
        CheckJson._get_metadata()

    def __repr__(self):
        """Return string representation."""
        return self.header + ': ' + str(self.exception)

    def print_traceback(self):
        """Print the traceback of the failure."""
        print(self.exception)

    @property
    def check_name(self):
        return self._check_name
