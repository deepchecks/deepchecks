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
from typing import Optional

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import HtmlSerializer

__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(HtmlSerializer['check_types.CheckFailure']):
    """Serializes any CheckFailure instance into html format.

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

    def serialize(
        self,
        full_html: bool = False,
        **kwargs
    ) -> str:
        """Serialize a CheckFailure instance into html format.

        Returns
        -------
        str
        """
        header = self.prepare_header()
        content = ''.join([header, self.prepare_summary(), self.prepare_error_message()])
        if full_html is True:
            return (
                '<html>'
                f'<head><title>{header}</title></head>'
                f'<body style="background-color: white;">{content}</body>'
                '</html>'
            )
        else:
            return content

    def prepare_header(self, output_id: Optional[str] = None) -> str:
        """Prepare the header section of the html output."""
        header = self.value.get_header()
        header = f'<b>{header}</b>'
        if output_id is not None:
            check_id = self.value.get_check_id(output_id)
            return f'<h4 id="{check_id}">{header}</h4>'
        else:
            return f'<h4>{header}</h4>'

    def prepare_summary(self) -> str:
        """Prepare the summary section of the html output."""
        return f'<p>{self.value.get_metadata()["summary"]}</p>'

    def prepare_error_message(self) -> str:
        """Prepare the error message of the html output."""
        return f'<p style="color:red">{self.value.exception}</p>'
