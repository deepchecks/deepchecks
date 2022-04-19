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
"""Module containing Wandb serializer for the SuiteResult type."""
import typing as t
from collections import OrderedDict

from wandb.sdk.data_types import WBValue

from deepchecks.core.suite import SuiteResult
from deepchecks.core.check_result import CheckResult
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import WandbSerializer
from deepchecks.core.presentation.check_result.wandb import CheckResultSerializer
from deepchecks.core.presentation.check_failure.wandb import CheckFailureSerializer


class SuiteResultSerializer(WandbSerializer[SuiteResult]):

    def __init__(self, value: SuiteResult, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        suite_name = self.value.name
        results: t.List[t.Tuple[str, WBValue]] = []

        for result in self.value.results:
            if isinstance(result, CheckResult):
                results.extend([
                    (f'{suite_name}/{k}', v)
                    for k, v in CheckResultSerializer(result).serialize().items()
                ])
            elif isinstance(result, CheckFailure):
                results.extend([
                    (f'{suite_name}/{k}', v)
                    for k, v in CheckFailureSerializer(result).serialize().items()
                ])
            else:
                raise TypeError(f'Unknown result type - {type(result)}')

        return OrderedDict(results)
