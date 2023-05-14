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

import pandas as pd


def read_and_save_data(assets_dir, file_name, url_to_file, to_numpy=True):
    """ If the file exsist reads it from the assets' directory, otherwise reads it from the url and saves it."""
    os.makedirs(assets_dir, exist_ok=True)
    if (assets_dir / file_name).exists():
        data = pd.read_csv(assets_dir / file_name, index_col=0)
    else:
        data = pd.read_csv(url_to_file, index_col=0)
        data.to_csv(assets_dir / file_name)

    if to_numpy:
        data = data.to_numpy()
    return data
