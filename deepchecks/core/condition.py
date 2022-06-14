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
"""Module containing all the base classes for checks."""
import enum
from typing import Callable, Dict, cast

from deepchecks.core.errors import DeepchecksValueError

__all__ = [
    'Condition',
    'ConditionResult',
    'ConditionCategory'
]


class Condition:
    """Contain condition attributes.

    Parameters
    ----------
    name : str
    function : Callable
    params : Dict

    """

    name: str
    function: Callable
    params: Dict

    def __init__(self, name: str, function: Callable, params: Dict):
        if not isinstance(function, Callable):
            raise DeepchecksValueError(f'Condition must be a function `(Any) -> Union[ConditionResult, bool]`, '
                                       f'but got: {type(function).__name__}')
        if not isinstance(name, str):
            raise DeepchecksValueError(f'Condition name must be of type str but got: {type(name).__name__}')
        self.name = name
        self.function = function
        self.params = params

    def __call__(self, *args, **kwargs) -> 'ConditionResult':
        """Run this condition."""
        result = cast(ConditionResult, self.function(*args, **kwargs))
        result.set_name(self.name)
        return result


class ConditionCategory(enum.Enum):
    """Condition result category. indicates whether the result should fail the suite."""

    FAIL = 'FAIL'
    WARN = 'WARN'
    PASS = 'PASS'
    ERROR = 'ERROR'


class ConditionResult:
    """Contain result of a condition function.

    Parameters
    ----------
    category : ConditionCategory
        The category to which the condition result belongs.
    details : str
        What actually happened in the condition.


    """

    category: ConditionCategory
    details: str
    name: str

    def __init__(self, category: ConditionCategory, details: str = ''):
        self.details = details
        self.category = category

    def set_name(self, name: str):
        """Set name to be displayed in table.

        Parameters
        ----------
        name : str
            Description of the condition to be displayed.
        """
        self.name = name

    @property
    def priority(self) -> int:
        """Return priority of the current condition.

        This value is primarily used to determine the order in which
        conditions should be displayed.

        Returns
        -------
        int
            condition priority value.
        """
        if self.category == ConditionCategory.PASS:
            return 4
        elif self.category == ConditionCategory.FAIL:
            return 1
        elif self.category == ConditionCategory.WARN:
            return 2
        return 3  # if error

    def is_pass(self, fail_if_warning=True) -> bool:
        """Return true if the condition has passed."""
        passed_categories = (
            (ConditionCategory.PASS,)
            if fail_if_warning
            else (ConditionCategory.PASS, ConditionCategory.WARN)
        )
        return self.category in passed_categories

    def get_icon(self):
        """Return icon of the result to display."""
        if self.category == ConditionCategory.PASS:
            return '<div style="color: green;text-align: center">\U00002713</div>'
        elif self.category == ConditionCategory.FAIL:
            return '<div style="color: red;text-align: center">\U00002716</div>'
        elif self.category == ConditionCategory.WARN:
            return '<div style="color: orange;text-align: center;font-weight:bold">\U00000021</div>'
        return '<div style="color: firebrick;text-align: center;font-weight:bold">\U00002048</div>'

    def __repr__(self):
        """Return string representation for printing."""
        return str(vars(self))
