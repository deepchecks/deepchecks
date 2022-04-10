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
