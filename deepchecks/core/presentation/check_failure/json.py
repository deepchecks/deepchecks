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

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import JsonSerializer


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(JsonSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[t.Any, t.Any]:
        return {
            'header': self.value.header,
            'check': self.value.check.metadata(),
        }