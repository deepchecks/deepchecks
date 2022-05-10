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
"""Module containing the integrations of the deepchecks.tabular package with the h2o autoML package."""

import pandas as pd
import numpy as np
try:
    import h2o
except ImportError as e:
    raise ImportError(
        'H2OWrapper requires the h2o python package. '
        'To get it, run "pip install h2o".'
    ) from e


class H2OWrapper:

    def __init__(self, h2o_model):
        self.model = h2o_model

    def predict(self, df: pd.DataFrame) -> np.array:
        return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 0]

    def predict_proba(self, df: pd.DataFrame) -> np.array:
        return self.model.predict(h2o.H2OFrame(df)).as_data_frame().values[:, 1:].astype(float)
