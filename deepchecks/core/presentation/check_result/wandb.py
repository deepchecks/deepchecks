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
from collections import OrderedDict

import pandas as pd
import wandb
from wandb.sdk.data_types import WBValue
from pandas.io.formats.style import Styler
from plotly.basedatatypes import BaseFigure

from deepchecks.core.check_result import CheckResult
from deepchecks.core.presentation.abc import WandbSerializer
from deepchecks.core.presentation.common import normalize_value
from deepchecks.core.presentation.common import aggregate_conditions
from deepchecks.core.presentation.common import pretify


__all__ = ['CheckResultSerializer']


class CheckResultSerializer(WandbSerializer[CheckResult]):

    def __init__(self, value: CheckResult, **kwargs):
        self.value = value

    def serialize(self, **kwargs) -> t.Dict[str, t.Any]:
        header = self.value.header
        output = OrderedDict()
        conditions_table = self.prepare_conditions_table()

        if conditions_table is not None:
            output[f'{header}/conditions table'] = conditions_table

        for section_name, wbvalue in self.prepare_display():
            output[f'{header}/{section_name}'] = wbvalue

        output[f'{header}/results'] = self.prepare_summary_table()

        return output

    def prepare_summary_table(self) -> wandb.Table:
        check_result = self.value
        metadata = check_result.check.metadata()
        return wandb.Table(
            columns=['header', 'params', 'summary', 'value'],
            data=[[
                check_result.header,
                pretify(metadata['params']),
                metadata['summary'],
                pretify(normalize_value(check_result.value))
            ]],
        )

    def prepare_conditions_table(self) -> t.Optional[wandb.Table]:
        if self.value.conditions_results:
            df = aggregate_conditions(self.value, include_icon=False)
            return wandb.Table(dataframe=df.data, allow_mixed_types=True)

    def prepare_display(self) -> t.Iterator[t.Tuple[str, WBValue]]:
        table_index = plot_index = html_index = 0

        for item in self.value.display:
            if isinstance(item, Styler):
                yield (
                    f'table {table_index}',
                    wandb.Table(dataframe=item.data.reset_index(), allow_mixed_types=True)
                )
                table_index += 1
            elif isinstance(item, pd.DataFrame):
                yield (
                    f'table {table_index}',
                    wandb.Table(dataframe=item.reset_index(), allow_mixed_types=True)
                )
                table_index += 1
            elif isinstance(item, str):
                yield (f'html {html_index}', wandb.Html(data=item))
                html_index += 1
            elif isinstance(item, BaseFigure):
                yield (f'plot {plot_index}', wandb.Plotly(item))
                plot_index += 1
            elif callable(item):
                raise NotImplementedError()
            else:
                raise TypeError(f'Unable to process display item of type: {type(item)}')
