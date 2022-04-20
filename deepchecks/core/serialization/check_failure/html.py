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
"""Module containing html serializer for the CheckFailuer type."""
from deepchecks.utils.strings import get_docs_summary
from deepchecks.core.check_result import CheckFailure
from deepchecks.core.serialization.abc import HtmlSerializer


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(HtmlSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> str:
        return ''.join([
            self.prepare_header(),
            self.prepare_summary(),
            self.prepare_error_message()
        ])

    def prepare_header(self) -> str:
        return f'<h4>{self.value.header}</h4>'

    def prepare_summary(self) -> str:
        return (
            f'<p>{get_docs_summary(self.value.check)}</p>'
            if hasattr(self.value.check, '__doc__')
            else ''
        )

    def prepare_error_message(self) -> str:
        return f'<p style="color:red"> {self.value.exception}</p>'