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
"""Module containing Wandb serializer for the CheckFailuer type."""
import typing as t

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import WandbSerializer
from deepchecks.core.serialization.common import prettify
from deepchecks.utils.wandb_utils import WANDB_INSTALLATION_CMD

try:
    import wandb
except ImportError as e:
    raise ImportError(
        'Wandb serializer requires the wandb python package. '
        f'To get it, run - {WANDB_INSTALLATION_CMD}.'
    ) from e


if t.TYPE_CHECKING:
    from wandb.sdk.data_types.base_types.wb_value import WBValue  # pylint: disable=unused-import


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WandbSerializer['check_types.CheckFailure']):
    """Serializes any CheckFailure instance into Wandb media format.

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

    def serialize(self, **kwargs) -> t.Dict[str, 'WBValue']:
        """Serialize a CheckFailure instance into Wandb media format.

        Returns
        -------
        Dict[str, WBValue]
        """
        header = self.value.header
        metadata = self.value.get_metadata()
        summary_table = wandb.Table(
            columns=['header', 'params', 'summary', 'value'],
            data=[[
                header,
                prettify(metadata['params']),
                metadata['summary'],
                str(self.value.exception)
            ]]
        )
        return {f'{header}/results': summary_table}
