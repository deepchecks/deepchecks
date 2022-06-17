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
"""Module containing ipywidget serializer for the CheckFailuer type."""
from ipywidgets import HTML, VBox

from deepchecks.core import check_result as check_types
from deepchecks.core.serialization.abc import WidgetSerializer
from deepchecks.core.serialization.common import normalize_widget_style

from . import html

__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WidgetSerializer['check_types.CheckFailure']):
    """Serializes any CheckFailure instance into an ipywidgets.Widget instance.

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
        self._html_serializer = html.CheckFailureSerializer(self.value)

    def serialize(self, **kwargs) -> VBox:
        """Serialize a CheckFailure instance into an ipywidgets.Widget instance.

        Returns
        -------
        ipywidgets.VBox
        """
        return normalize_widget_style(VBox(children=(
            self.prepare_header(),
            self.prepare_summary(),
            self.prepare_error_message()
        )))

    def prepare_header(self) -> HTML:
        """Prepare header widget.

        Returns
        -------
        ipywidgets.HTML
        """
        return HTML(value=self._html_serializer.prepare_header())

    def prepare_summary(self) -> HTML:
        """Prepare header summary.

        Returns
        -------
        ipywidgets.HTML
        """
        return HTML(value=self._html_serializer.prepare_summary())

    def prepare_error_message(self) -> HTML:
        """Prepare error message widget.

        Returns
        -------
        ipywidgets.HTML
        """
        return HTML(value=self._html_serializer.prepare_error_message())
