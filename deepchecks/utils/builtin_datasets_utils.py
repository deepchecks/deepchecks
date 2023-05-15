# ----------------------------------------------------------------------------
# Copyright (C) 2021-2023 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""utils for loading saving and utilizing deepchecks built in datasets."""
import os
from io import BytesIO

import numpy as np
import pandas as pd
import requests


def read_and_save_data(assets_dir, file_name, url_to_file, file_type='csv', to_numpy=False):
    """If the file exist reads it from the assets' directory, otherwise reads it from the url and saves it."""
    os.makedirs(assets_dir, exist_ok=True)
    if (assets_dir / file_name).exists():
        if file_type == 'csv':
            data = pd.read_csv(assets_dir / file_name, index_col=0)
        elif file_type == 'npy':
            data = np.load(assets_dir / file_name)
        else:
            raise ValueError('file_type must be either "csv" or "npy"')
    else:
        if file_type == 'csv':
            data = pd.read_csv(url_to_file, index_col=0)
            data.to_csv(assets_dir / file_name)
        elif file_type == 'npy':
            data = np.load(BytesIO(requests.get(url_to_file).content))
            np.save(assets_dir / file_name, data)
        else:
            raise ValueError('file_type must be either "csv" or "npy"')

    if to_numpy:
        if isinstance(data, pd.DataFrame):
            data = data.to_numpy()
        elif not isinstance(data, np.ndarray):
            raise ValueError(f'Unknown data type - {type(data)}. Must be either pandas.DataFrame or numpy.ndarray')
    return data
