# ----------------------------------------------------------------------------
# Copyright (C) 2021 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
from pathlib import Path
from runpy import run_path
import glob

import torch


def test_plots_on_gpu():
    """If there is GPU available running all the docs plot files. Only makes sure the plots don't crash, and not \
    testing any other display or functionality."""
    if torch.cuda.is_available():
        path = Path(__file__).parent.parent.parent.parent / "docs" / "source" / "**" / "plot_*.py"
        plots_files = glob.glob(str(path), recursive=True)
        if not plots_files:
            raise ValueError("No plots found in docs/source")
        for file in plots_files:
            run_path(file)
