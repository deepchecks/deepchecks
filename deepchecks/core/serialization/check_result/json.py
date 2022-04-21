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
import typing as t

import pandas as pd
import jsonpickle.ext.pandas as jsonpickle_pd
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure
from typing_extensions import TypedDict

from deepchecks.core.check_result import CheckResult
from deepchecks.core.checks import CheckMetadata
from deepchecks.core.serialization.abc import JsonSerializer
from deepchecks.core.serialization.abc import ABCDisplayItemsHandler
from deepchecks.core.serialization.common import aggregate_conditions
from deepchecks.core.serialization.common import normalize_value
# from deepchecks.core.serialization.common import pretify


# registers jsonpickle pandas extension for pandas support in the to_json function
jsonpickle_pd.register_handlers()


__all__ = ['CheckResultSerializer']


class CheckResultMetadata(TypedDict):
    check: CheckMetadata
    value: t.Any
    header: str
    conditions_results: t.List[t.Dict[t.Any, t.Any]]
    display: t.List[t.Any]


class CheckResultSerializer(JsonSerializer[CheckResult]):
    """Serializes any CheckResult instance into JSON format.

    Parameters
    ----------
    value : CheckResult
        CheckResult instance that needed to be serialized.
    """

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> CheckResultMetadata:
        """Serialize a CheckResult instance into JSON format.

        Returns
        -------
        CheckResultMetadata
        """
        return CheckResultMetadata(
            check=self.prepare_check_metadata(),
            header=self.value.get_header(),
            value=self.prepare_value(),
            conditions_results=self.prepare_condition_results(),
            display=self.prepare_display()
        )

    def prepare_check_metadata(self) -> CheckMetadata:
        """Prepare Check instance metadata dictionary."""
        return self.value.check.metadata(with_doc_link=True)

    def prepare_condition_results(self) -> t.List[t.Dict[t.Any, t.Any]]:
        """Serialize condition results into json."""
        if self.value.have_conditions:
            df = aggregate_conditions(self.value, include_icon=False)
            return df.data.to_json(orient='records')
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
                'payload': item.data.to_json(orient='records')
            }
        else:
            return {
                'type': 'dataframe',
                'payload': item.to_json(orient='records')
            }

    @classmethod
    def handle_callable(cls, item: t.Callable, index: int, **kwargs) -> str:
        """Handle callable."""
        raise NotImplementedError()

    @classmethod
    def handle_figure(cls, item: BaseFigure, index: int, **kwargs) -> t.Dict[str, t.Any]:
        """Handle plotly figure item."""
        return {'type': 'plotly', 'payload': item.to_json()}
