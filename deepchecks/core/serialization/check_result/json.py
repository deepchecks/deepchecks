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
from deepchecks.core.serialization.common import aggregate_conditions
from deepchecks.core.serialization.common import normalize_value
from deepchecks.core.serialization.common import pretify


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

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> CheckResultMetadata:
        return CheckResultMetadata(
            check=self.prepare_check_metadata(),
            header=self.value.get_header(),
            value=self.prepare_value(),
            conditions_results=self.prepare_condition_results(),
            display=self.prepare_display()
        )

    def prepare_check_metadata(self) -> CheckMetadata:
        return self.value.check.metadata(with_doc_link=True)

    def prepare_condition_results(self) -> t.List[t.Dict[t.Any, t.Any]]:
        if self.value.have_conditions:
            df = aggregate_conditions(self.value, include_icon=False)
            return df.data.to_json(orient='records')
        else:
            return []

    def prepare_value(self) -> str:
        return pretify(normalize_value(self.value.value))

    def prepare_display(self) -> t.List[t.Any]:
        output = []
        for item in self.value.display:
            if isinstance(item, Styler):
                output.append({
                    'type': 'dataframe',
                    'payload': item.data.to_json(orient='records')
                })
            elif isinstance(item, pd.DataFrame):
                output.append({
                    'type': 'dataframe',
                    'payload': item.to_json(orient='records')
                })
            elif isinstance(item, str):
                output.append({'type': 'html', 'payload': item})
            elif isinstance(item, BaseFigure):
                output.append({'type': 'plotly', 'payload': item.to_json()})
            elif callable(item):
                raise NotImplementedError
            else:
                raise TypeError(f'Unable to serialize into json item of type - {type(item)}')
        return output