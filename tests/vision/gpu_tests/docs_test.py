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
import time
from pathlib import Path
from runpy import run_path

import torch
import wandb

# Since we have a plot that's calling `wandb.login` we need to setup first
wandb.setup(wandb.Settings(mode="disabled", program=__name__, program_relpath=__name__, disable_code=True))

DOCS_EXAMPLES_DIR = ["checks/vision",
                     "checks/tabular",
                     "user-guide/tabular/tutorials",
                     "user-guide/vision/tutorials",
                     "user-guide/general/customizations",
                     "user-guide/general/exporting_results", ]


def test_plots_on_gpu():
    """If there is GPU available running all the docs plot files. Only makes sure the plots don't crash, and not \
    testing any other display or functionality."""
    if torch.cuda.is_available():
        path = Path(__file__).parent.parent.parent.parent / "docs" / "source"
        # Take only source file and excluding compiled files
        source_files = set()
        for folder in DOCS_EXAMPLES_DIR:
            source_files.update(set(path.glob(f"**/{folder}/**/plot_*.py")))

        if not source_files:
            raise ValueError("No plots found in docs/source")
        for file in source_files:
            print(f"plot file: {str(file)}")
            start = time.time()
            run_path(str(file))
            end = time.time()
            print(f"plot file: {str(file)}, Time: {end - start}")
