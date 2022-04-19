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
from ipywidgets import VBox, HTML

from deepchecks.core.check_result import CheckFailure
from deepchecks.core.presentation.abc import WidgetSerializer

from . import html


__all__ = ['CheckFailureSerializer']


class CheckFailureSerializer(WidgetSerializer[CheckFailure]):

    def __init__(self, value: CheckFailure, **kwargs):
        self.value = value
        self._html_serializer = html.CheckFailureSerializer(self.value)

    def serialize(self, **kwargs) -> VBox:
        return VBox(children=(
            self.prepare_header(),
            self.prepare_summary(),
            self.prepare_error_message()
        ))

    def prepare_header(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_header())

    def prepare_summary(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_summary())

    def prepare_error_message(self) -> HTML:
        return HTML(value=self._html_serializer.prepare_error_message())
