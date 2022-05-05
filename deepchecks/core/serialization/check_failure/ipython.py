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
""""""
import typing as t
from IPython.display import HTML

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import IPythonSerializer, IPythonDisplayable

from . import html


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(IPythonSerializer['check_types.CheckFailure']):

    def __init__(self, value: 'check_types.CheckFailure', **kwargs):
        self.value = value
        self._html_serializer = html.CheckFailureSerializer(value)

    def serialize(self, **kwargs) -> t.List[IPythonDisplayable]:
        return [HTML(self._html_serializer.serialize())]