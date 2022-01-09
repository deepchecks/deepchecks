# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Module containing all the base classes for checks."""
# pylint: disable=broad-except
import enum
from typing import Callable, Dict, cast

from deepchecks.errors import DeepchecksValueError


__all__ = [
    'Condition',
    'ConditionResult',
    'ConditionCategory'
]


class Condition:
    """Contain condition attributes."""

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


class ConditionResult:
    """Contain result of a condition function."""

    is_pass: bool
    category: ConditionCategory
    details: str
    name: str

    def __init__(self, is_pass: bool, details: str = '',
                 category: ConditionCategory = ConditionCategory.FAIL):
        """Initialize condition result.

        Args:
            is_pass (bool): Whether the condition functions passed the given value or not.
            details (str): What actually happened in the condition.
            category (ConditionCategory): The category to which the condition result belongs.
        """
        self.is_pass = is_pass
        self.details = details
        self.category = category

    def set_name(self, name: str):
        """Set name to be displayed in table.

        Args:
            name (str): Description of the condition to be displayed.
        """
        self.name = name

    @property
    def priority(self) -> int:
        """Return priority of the current condition.

        This value is primarily used to determine the order in which
        conditions should be displayed.

        Returns:
            int: condition priority value;
        """
        if self.is_pass is True:
            return 3
        elif self.category == ConditionCategory.FAIL:
            return 1
        else:
            return 2

    def get_icon(self):
        """Return icon of the result to display."""
        if self.is_pass:
            return '<div style="color: green;text-align: center">\U00002713</div>'
        elif self.category == ConditionCategory.FAIL:
            return '<div style="color: red;text-align: center">\U00002716</div>'
        else:
            return '<div style="color: orange;text-align: center;font-weight:bold">\U00000021</div>'

    def __repr__(self):
        """Return string representation for printing."""
        return str(vars(self))
