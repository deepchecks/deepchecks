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
"""Module containing json serializer for the CheckFailuer type."""
import typing as t

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import JsonSerializer

__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(JsonSerializer['check_types.CheckFailure']):
    """Serializes any CheckFailure instance into JSON format.

    Parameters
    ----------
    value : CheckFailure
        CheckFailure instance that needed to be serialized.
    """

    def __init__(self, value: 'check_types.CheckFailure', **kwargs):
        if not isinstance(value, check_types.CheckFailure):
            raise TypeError(
                f'Expected "CheckFailure" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(self, **kwargs) -> t.Dict[str, t.Any]:
        """Serialize a CheckFailure instance into JSON format.

        Returns
        -------
        Dict[str, Any]
        """
        return {
            'header': self.value.header,
            'type': 'CheckFailure',
            'check': self.value.check.metadata(),
            'exception': str(self.value.exception),
        }
