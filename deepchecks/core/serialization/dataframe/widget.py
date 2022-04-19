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
"""Module containing widget serializer for the pandas.DataFrame type."""
import typing as t

import pandas as pd
from pandas.io.formats.style import Styler
from ipywidgets import HTML

from deepchecks.core.serialization.abc import WidgetSerializer
from . import html


__all__ = ['DataFrameSerializer']


DataFrameOrStyler = t.Union[pd.DataFrame, Styler]


class DataFrameSerializer(WidgetSerializer[DataFrameOrStyler]):

    def __init__(self, value: DataFrameOrStyler, **kwargs):
        self.value = value
        self._html_serializer = html.DataFrameSerializer(self.value)

    def serialize(self, **kwargs) -> HTML:
        return HTML(value=self._html_serializer.serialize())
