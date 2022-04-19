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
from ipywidgets import VBox
from deepchecks.core.suite import SuiteResult
from deepchecks.core.presentation.abc import Presentation
from . import html, widget


__all__ = ['SuiteResultPresentation']


class SuiteResultPresentation(Presentation[SuiteResult]):

    def __init__(self, value: SuiteResult, **kwargs):
        self.value = value
        self._html_serializer = html.SuiteResultSerializer(self.value)
        self._widget_serializer = widget.SuiteResultSerializer(self.value)

    def to_html(self, **kwargs) -> str:
        return self._html_serializer.serialize(**kwargs)

    def to_widget(self, **kwargs) -> VBox:
        return self._widget_serializer.serialize(**kwargs)