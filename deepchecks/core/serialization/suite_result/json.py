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
"""Module containing JSON serializer for the SuiteResult type."""
import typing as t

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import JsonSerializer
from deepchecks.core.serialization.check_failure.json import CheckFailureSerializer
from deepchecks.core.serialization.check_result.json import CheckResultSerializer

__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(JsonSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into JSON format.

    Parameters
    ----------
    value : SuiteResult
        SuiteResult instance that needed to be serialized.
    """

    def __init__(self, value: 'suite.SuiteResult', **kwargs):
        if not isinstance(value, suite.SuiteResult):
            raise TypeError(
                f'Expected "SuiteResult" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(
        self,
        with_display: bool = True,
        **kwargs
    ) -> t.Union[t.Dict[t.Any, t.Any], t.List[t.Any]]:
        """Serialize a SuiteResult instance into JSON format.

        Parameters
        ----------
        with_display : bool, default True
            whether to include serialized `CheckResult.display` items into
            the output or not
        **kwargs :
            all other key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        Union[Dict[Any, Any], List[Any]]
        """
        results = []

        for it in self.value.results:
            if isinstance(it, check_types.CheckResult):
                results.append(CheckResultSerializer(it).serialize(with_display=with_display))
            elif isinstance(it, check_types.CheckFailure):
                results.append(CheckFailureSerializer(it).serialize())
            else:
                raise TypeError(f'Unknown result type - {type(it)}')

        return {'name': self.value.name, 'results': results, 'type' : 'SuiteResult'}
