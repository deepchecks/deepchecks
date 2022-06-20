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

from deepchecks.core import check_result as check_types
from deepchecks.core import suite
from deepchecks.core.serialization.abc import WandbSerializer
from deepchecks.core.serialization.check_failure.wandb import CheckFailureSerializer
from deepchecks.core.serialization.check_result.wandb import CheckResultSerializer

if t.TYPE_CHECKING:
    from wandb.sdk.data_types.base_types.wb_value import WBValue  # pylint: disable=unused-import


class SuiteResultSerializer(WandbSerializer['suite.SuiteResult']):
    """Serializes any SuiteResult instance into Wandb media format.

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

    def serialize(self, **kwargs) -> t.Dict[str, 'WBValue']:
        """Serialize a SuiteResult instance into Wandb media format.

        Parameters
        ----------
        **kwargs :
            all key-value arguments will be passed to the CheckResult/CheckFailure
            serializers

        Returns
        -------
        Dict[str, WBValue]
        """
        suite_name = self.value.name
        results: t.List[t.Tuple[str, 'WBValue']] = []

        for result in self.value.results:
            if isinstance(result, check_types.CheckResult):
                results.extend([
                    (f'{suite_name}/{k}', v)
                    for k, v in CheckResultSerializer(result).serialize(**kwargs).items()
                ])
            elif isinstance(result, check_types.CheckFailure):
                results.extend([
                    (f'{suite_name}/{k}', v)
                    for k, v in CheckFailureSerializer(result).serialize(**kwargs).items()
                ])
            else:
                raise TypeError(f'Unknown result type - {type(result)}')

        return OrderedDict(results)
