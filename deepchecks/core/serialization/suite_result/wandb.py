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

from deepchecks.core.suite import SuiteResult
from deepchecks.core.check_result import CheckResult
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.serialization.abc import WandbSerializer
from deepchecks.core.serialization.check_result.wandb import CheckResultSerializer
from deepchecks.core.serialization.check_failure.wandb import CheckFailureSerializer


try:
    from wandb.sdk.data_types import WBValue
except ImportError:
    raise ImportError(
        'Wandb serializer requires the wandb python package. '
        'To get it, run "pip install wandb".'
    )


class SuiteResultSerializer(WandbSerializer[SuiteResult]):
    """Serializes any SuiteResult instance into Wandb media format.

    Parameters
    ----------
    value : SuiteResult
        SuiteResult instance that needed to be serialized.
    """

    def __init__(self, value: SuiteResult, **kwargs):
        if isinstance(value, SuiteResult):
            raise TypeError(
                f'Expected "SuiteResult" but got "{type(value).__name__}"'
            )
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        """Serialize a SuiteResult instance into Wandb media format.

        Returns
        -------
        Dict[str, WBValue]
        """
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
