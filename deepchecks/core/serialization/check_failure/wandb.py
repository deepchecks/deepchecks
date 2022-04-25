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

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.serialization.abc import WandbSerializer
from deepchecks.core.serialization.common import prettify


try:
    import wandb
    from wandb.sdk.data_types.base_types.wb_value import WBValue
except ImportError:
    raise ImportError(
        'Wandb serializer requires the wandb python package. '
        'To get it, run "pip install wandb".'
    )


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WandbSerializer[CheckFailure]):
    """Serializes any CheckFailure instance into Wandb media format.

    Parameters
    ----------
    value : CheckFailure
        CheckFailure instance that needed to be serialized.
    """

    def __init__(self, value: CheckFailure, **kwargs):
        if not isinstance(value, CheckFailure):
            raise TypeError(
                f'Expected "CheckFailure" but got "{type(value).__name__}"'
            )
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        """Serialize a CheckFailure instance into Wandb media format.

        Returns
        -------
        Dict[str, WBValue]
        """
        header = self.value.header
        metadata = self.value.check.metadata()
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
