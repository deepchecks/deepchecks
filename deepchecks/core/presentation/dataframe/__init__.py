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
import typing as t
import pandas as pd
from ipywidgets import HTML
from pandas.io.formats.style import Styler
from deepchecks.core.presentation.abc import Presentation
from . import html


__all__ = ['DataFramePresentation']


class DataFramePresentation(Presentation[t.Union[pd.DataFrame, Styler]]):

    def __init__(self, value: t.Union[pd.DataFrame, Styler], **kwargs):
        self.value = value
        self._html_serializer = html.DataFrameSerializer(self.value)

    def to_html(self, **kwargs) -> str:
        return self._html_serializer.serialize(**kwargs)

    def to_widget(self, **kwargs) -> HTML:
        return HTML(value=self.to_html(**kwargs))
