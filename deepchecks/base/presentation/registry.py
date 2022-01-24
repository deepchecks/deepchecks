import typing as t

import pandas as pd
from pandas.io.formats.style import Styler
from typing_extensions import Literal

from deepchecks.base import ConditionResult
from deepchecks.base import CheckResult
from deepchecks.base import SuiteResult
from deepchecks.base.presentation.abc import Presentation


__register__ = {}


T = t.TypeVar("T")


@t.overload
def default_presentation(
    for_type: t.Type[T], 
    raise_error: Literal[True] = True
) -> t.Type[Presentation[T]]:
    ...

@t.overload
def default_presentation(
    for_type: t.Type[T], 
    raise_error: Literal[False] = False
) -> t.Optional[t.Type[Presentation[T]]]:
    ...

def default_presentation(
    for_type: t.Type[T], 
    raise_error: bool = True
) -> t.Optional[t.Type[Presentation[T]]]:
    # keeping it here to omit circular import problems
    from .dataframe import DataFramePresentation
    from .check_result import CheckResultPresentation
    from .condition_result import ConditionResultPresentation
    from .suite_result import SuiteResultPresentation

    presentations = {
        pd.DataFrame: DataFramePresentation,
        Styler: DataFramePresentation,
        t.Union[pd.DataFrame, Styler]: DataFramePresentation,
        ConditionResult: ConditionResultPresentation,
        t.List[ConditionResult]: ConditionResultPresentation,
        t.Union[ConditionResult, t.List[ConditionResult]]: ConditionResultPresentation,
        CheckResult: CheckResultPresentation,
        SuiteResult: SuiteResultPresentation
    }

    presentations.update(__register__)

    if raise_error is True and for_type not in presentations:
        raise KeyError(f"Did not find presentation for the type - {for_type}")
    
    if for_type not in presentations:
        return
    
    return presentations[for_type]


def add_presentation(for_type: T, impl: Presentation[T]):
    global __register__
    __register__[for_type] = impl

