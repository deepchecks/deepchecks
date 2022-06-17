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
"""Module containing json serializer for the CheckResult type."""
import base64
import typing as t

import pandas as pd
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from typing_extensions import TypedDict

from deepchecks.core import check_result as check_types
from deepchecks.core import checks  # pylint: disable=unused-import
from deepchecks.core.serialization.abc import ABCDisplayItemsHandler, JsonSerializer
from deepchecks.core.serialization.common import aggregate_conditions, normalize_value

__all__ = ['CheckResultSerializer']


class CheckResultMetadata(TypedDict):
    type: str
    check: 'checks.CheckMetadata'
    value: t.Any
    header: str
    conditions_results: t.List[t.Dict[t.Any, t.Any]]
    display: t.Optional[t.List[t.Any]]


class CheckResultSerializer(JsonSerializer['check_types.CheckResult']):
    """Serializes any CheckResult instance into JSON format.

    Parameters
    ----------
    value : CheckResult
        CheckResult instance that needed to be serialized.
    """

    def __init__(self, value: 'check_types.CheckResult', **kwargs):
        if not isinstance(value, check_types.CheckResult):
            raise TypeError(
                f'Expected "CheckResult" but got "{type(value).__name__}"'
            )
        super().__init__(value=value)

    def serialize(self, with_display: bool = True, **kwargs) -> CheckResultMetadata:
        """Serialize a CheckResult instance into JSON format.

        Parameters
        ----------
        with_display : bool
            controls if to serialize display or not

        Returns
        -------
        CheckResultMetadata
        """
        display = self.prepare_display() if with_display else None
        return CheckResultMetadata(
            type='CheckResult',
            check=self.prepare_check_metadata(),
            header=self.value.get_header(),
            value=self.prepare_value(),
            conditions_results=self.prepare_condition_results(),
            display=display
        )

    def prepare_check_metadata(self) -> 'checks.CheckMetadata':
        """Prepare Check instance metadata dictionary."""
        assert self.value.check is not None
        return self.value.check.metadata(with_doc_link=True)

    def prepare_condition_results(self) -> t.List[t.Dict[t.Any, t.Any]]:
        """Serialize condition results into json."""
        if self.value.have_conditions:
            df = aggregate_conditions(self.value, include_icon=False)
            return df.data.to_dict(orient='records')
        else:
            return []

    def prepare_value(self) -> t.Any:
        """Serialize CheckResult value var into JSON."""
        return normalize_value(self.value.value)

    def prepare_display(self) -> t.List[t.Dict[str, t.Any]]:
        """Serialize CheckResult display items into JSON."""
        return DisplayItemsHandler.handle_display(self.value.display)


class DisplayItemsHandler(ABCDisplayItemsHandler):
    """Auxiliary class to decouple display handling logic from other functionality."""

    @classmethod
    def handle_string(cls, item: str, index: int, **kwargs) -> t.Dict[str, str]:
        """Handle textual item."""
        return {'type': 'html', 'payload': item}

    @classmethod
    def handle_dataframe(
        cls,
        item: t.Union[pd.DataFrame, Styler],
        index: int,
        **kwargs
    ) -> t.Dict[str, t.Any]:
        """Handle dataframe item."""
        if isinstance(item, Styler):
            return {
                'type': 'dataframe',
                'payload': item.data.to_dict(orient='records')
            }
        else:
            return {
                'type': 'dataframe',
                'payload': item.to_dict(orient='records')
            }

    @classmethod
    def handle_callable(cls, item: t.Callable, index: int, **kwargs) -> t.Dict[str, t.Any]:
        """Handle callable."""
        return {
            'type': 'images',
            'payload': [
                base64.b64encode(buffer.read()).decode('ascii')
                for buffer in super().handle_callable(item, index, **kwargs)
            ]
        }

    @classmethod
    def handle_figure(cls, item: BaseFigure, index: int, **kwargs) -> t.Dict[str, t.Any]:
        """Handle plotly figure item."""
        return {'type': 'plotly', 'payload': item.to_json()}

    @classmethod
    def handle_display_map(cls, item: 'check_types.DisplayMap', index: int, **kwargs) -> t.Dict[str, t.Any]:
        """Handle display map instance item."""
        return {
            'type': 'displaymap',
            'payload': {
                k: cls.handle_display(v, **kwargs)
                for k, v in item.items()
            }
        }
