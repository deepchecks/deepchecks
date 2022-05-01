"""utils for testing."""
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
from typing import Callable

from torch.utils.data import DataLoader

from deepchecks.vision import VisionData


def get_modified_dataloader(vision_data: VisionData, func_to_apply: Callable, shuffle: bool = False) -> DataLoader:
    """Get a dataloader whose underlying dataset is modified by a function.

    Parameters
    ----------
    vision_data: VisionData
        A vision data object of the type for which the modified dataloader is intended.
    func_to_apply: Callable
        A callable of the form func_to_apply(orig_dataset, idx) that returns a modified version of the original
        dataset return value for the given index.
    shuffle: bool, default: False
        Whether return d dataloader with shuffling.

    Returns
    -------
    DataLoader
        The modified dataloader.
    """

    class ModifiedDataset():
        """A modified dataset object, returning func_to_apply for each index."""

        def __init__(self, orig_dataset):
            self._orig_dataset = orig_dataset

        def __getitem__(self, idx):
            return func_to_apply(self._orig_dataset, idx)

        def __len__(self):
            return len(self._orig_dataset)

    # Code needed to return a dataloader with the modified dataset that is otherwise identical to the original.
    props = vision_data._get_data_loader_props(vision_data.data_loader)  # pylint: disable=protected-access
    props['dataset'] = ModifiedDataset(vision_data.data_loader.dataset)
    props['shuffle'] = shuffle
    data_loader = DataLoader(**props)
    data_loader, _ = vision_data._get_data_loader_sequential(data_loader)  # pylint: disable=protected-access
    return data_loader
