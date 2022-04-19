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

import wandb
from wandb.sdk.data_types import WBValue

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import WandbSerializer
from deepchecks.core.presentation.common import pretify


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WandbSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[str, WBValue]:
        header = self.value.header
        metadata = self.value.check.metadata()
        summary_table = wandb.Table(
            columns=['header', 'params', 'summary', 'value'],
            data=[[
                header,
                pretify(metadata['params']),
                metadata['summary'],
                str(self.value.exception)
            ]]
        )
        return {f'{header}/results': summary_table}