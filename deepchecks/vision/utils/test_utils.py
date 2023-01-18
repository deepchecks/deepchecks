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
import random
from copy import copy
from hashlib import md5
from typing import Dict, Iterator, List, Sized

import numpy as np
import torch
from PIL import Image
from torch.utils.data import BatchSampler, DataLoader, Sampler

from deepchecks.vision import VisionData
from deepchecks.vision.vision_data.utils import object_to_numpy


def replace_collate_fn_visiondata(vision_data: VisionData, new_collate_fn):
    """Create a new VisionData based on the same attributes as the old one with updated collate function."""
    new_data_loader = replace_collate_fn_dataloader(vision_data.batch_loader, new_collate_fn)
    return VisionData(new_data_loader, task_type=vision_data.task_type.value, reshuffle_data=False,
                      dataset_name=vision_data.name,
                      label_map=vision_data.label_map)


def replace_collate_fn_dataloader(data_loader: DataLoader, new_collate_fn):
    """Replace collate_fn function in DataLoader."""
    props = _get_data_loader_props(data_loader)
    _collisions_removal_dataloader_props(props)
    props['collate_fn'] = new_collate_fn
    return data_loader.__class__(**props)


def _collisions_removal_dataloader_props(props: Dict):
    if 'batch_sampler' in props:
        for key in ['batch_size', 'shuffle', 'sampler', 'drop_last']:
            props.pop(key, None)


def _get_data_loader_props(data_loader: DataLoader) -> Dict:
    """Get properties relevant for the copy of a DataLoader."""
    attr_list = ['num_workers',
                 'collate_fn',
                 'pin_memory',
                 'timeout',
                 'worker_init_fn',
                 'prefetch_factor',
                 'persistent_workers',
                 'batch_size',
                 'shuffle',
                 'generator',
                 'sampler',
                 'batch_sampler']
    aval_attr = {}
    for attr in attr_list:
        if hasattr(data_loader, attr):
            aval_attr[attr] = getattr(data_loader, attr)
    aval_attr['dataset'] = copy(data_loader.dataset)
    return aval_attr


def get_data_loader_sequential(data_loader: DataLoader, shuffle: bool = False, n_samples=None) -> DataLoader:
    """Create new DataLoader with sampler of type IndicesSequentialSampler. This makes the data loader have \
    consistent batches order."""
    # First set generator seed to make it reproducible
    if data_loader.generator:
        data_loader.generator.set_state(torch.Generator().manual_seed(42).get_state())
    indices = []
    batch_sampler = data_loader.batch_sampler
    # Using the batch sampler to get all indices
    for batch in batch_sampler:
        indices += batch
    if shuffle:
        indices = random.sample(indices, len(indices))
    if n_samples is not None:
        indices = indices[:n_samples]

    # Create new sampler and batch sampler
    sampler = IndicesSequentialSampler(indices)
    new_batch_sampler = BatchSampler(sampler, batch_sampler.batch_size, batch_sampler.drop_last)

    props = _get_data_loader_props(data_loader)
    props['batch_sampler'] = new_batch_sampler
    _collisions_removal_dataloader_props(props)
    return data_loader.__class__(**props)


class IndicesSequentialSampler(Sampler):
    """Samples elements sequentially from a given list of indices, without replacement.
    Args:
        indices (sequence): a sequence of indices
    """

    indices: List[int]

    def __init__(self, indices: List[int]) -> None:
        super().__init__(None)
        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        """Return an iterator over the indices."""
        return iter(self.indices)

    def __len__(self) -> int:
        """Return the number of indices."""
        return len(self.indices)

    def index_at(self, location):
        """Return for a given location, the real index value."""
        return self.indices[location]


def un_normalize_batch(tensor: torch.Tensor, mean: Sized, std: Sized, max_pixel_value: int = 255):
    """Apply un-normalization on a tensor in order to display an image."""
    dim = len(mean)
    reshape_shape = (1, 1, 1, dim)
    max_pixel_value = [max_pixel_value] * dim
    mean = torch.tensor(mean, device=tensor.device).reshape(reshape_shape)
    std = torch.tensor(std, device=tensor.device).reshape(reshape_shape)
    tensor = (tensor * std) + mean
    tensor = tensor * torch.tensor(max_pixel_value, device=tensor.device).reshape(reshape_shape)
    return object_to_numpy(tensor)


def hash_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image.squeeze())
    elif isinstance(image, torch.Tensor):
        image = Image.fromarray(object_to_numpy(image).squeeze())

    image = image.resize((10, 10))
    image = image.convert('L')

    pixel_data = list(image.getdata())
    avg_pixel = sum(pixel_data) / len(pixel_data)

    bits = ''.join(['1' if (px >= avg_pixel) else '0' for px in pixel_data])
    hex_representation = str(hex(int(bits, 2)))[2:][::-1].upper()
    md_5hash = md5()
    md_5hash.update(hex_representation.encode())
    return md_5hash.hexdigest()
