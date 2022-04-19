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
import typing as t

from deepchecks.core.suite import SuiteResult
from deepchecks.core.check_result import CheckResult
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import JsonSerializer
from deepchecks.core.presentation.check_result.json import CheckResultSerializer
from deepchecks.core.presentation.check_failure.json import CheckFailureSerializer


__all__ = ['SuiteResultSerializer']


class SuiteResultSerializer(JsonSerializer[SuiteResult]):

    def __init__(self, value: SuiteResult, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> t.Union[t.Dict[t.Any, t.Any], t.List[t.Any]]:
        results = []

        for it in self.value.results:
            if isinstance(it, CheckResult):
                results.append(CheckResultSerializer(it).serialize())
            elif isinstance(it, CheckFailure):
                results.append(CheckFailureSerializer(it).serialize())
            else:
                raise TypeError(f'Unknown result type - {type(it)}')

        return {'name': self.value.name, 'results': results}
