import sys
import typing as t
import numpy as np
import pandas as pd

from deepchecks.tabular.checks import model_evaluation

__all__ = model_evaluation.__all__

this_module = sys.modules[__name__]

for check_name in __all__:
    check_class = getattr(model_evaluation, check_name)

    class RecCheck(check_class):
         run = ...
    setattr(this_module, check_name, RecCheck)
