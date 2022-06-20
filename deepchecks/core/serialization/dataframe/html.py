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
"""Module containing html serializer for the pandas.DataFrame type."""
import typing as t
import warnings

import pandas as pd
from pandas.io.formats.style import Styler

from deepchecks.core.serialization.abc import HtmlSerializer

__all__ = ['DataFrameSerializer']


DataFrameOrStyler = t.Union[pd.DataFrame, Styler]


class DataFrameSerializer(HtmlSerializer[DataFrameOrStyler]):
    """Serializes pandas.DataFrame instance into HTML format.

    Parameters
    ----------
    value : Union[pandas.DataFrame, Styler]
        DataFrame instance that needed to be serialized.
    """

    def __init__(self, value: DataFrameOrStyler, **kwargs):
        if not isinstance(value, (pd.DataFrame, Styler)):
            raise TypeError(
                f'Expected "Union[DataFrame, Styler]" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(self, classes: t.Optional[str] = None, **kwargs) -> str:
        """Serialize pandas.DataFrame instance into HTML format."""
        table_attributes = f'class="{classes}"' if classes is not None else ''
        try:
            if isinstance(self.value, pd.DataFrame):
                df_styler = self.value.style
            else:
                df_styler = self.value
            # Using deprecated pandas method so hiding the warning
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                df_styler.set_precision(2)
                table_css_props = (
                    ('text-align', 'left'),  # Align everything to the left
                    ('white-space', 'pre-wrap')  # Define how to handle white space characters (like \n)
                )
                df_styler.set_table_styles([
                    dict(selector='table,thead,tbody,th,td', props=table_css_props),
                    dict(selector='table', props=(('width', 'max-content'),)),
                ])
                return df_styler.render(table_attributes=table_attributes)
        # Because of MLC-154. Dataframe with Multi-index or non unique indices does not have a style
        # attribute, hence we need to display as a regular pd html format.
        except ValueError:
            return self.value.to_html(classes=classes)
