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
"""module contains the Identifier Leakage check - deprecated."""

import warnings
from typing import Callable, Dict, List, Union

from deepchecks.tabular.checks.model_evaluation import TrainTestPerformance


class PerformanceReport(TrainTestPerformance):
    """Deprecated. Summarize given scores on a dataset and model."""

    def __init__(self,
                 scorers: Union[List[str], Dict[str, Union[str, Callable]]] = None,
                 reduce: Union[Callable, str] = 'mean',
                 **kwargs):
        warnings.warn('the performance report check is deprecated. use the train test performance check instead',
                      DeprecationWarning, stacklevel=2)
        TrainTestPerformance.__init__(self, scorers, reduce, **kwargs)
