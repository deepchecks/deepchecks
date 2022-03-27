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
import pathlib
from shutil import rmtree

import numpy as np
import pytest
import PIL.Image as pilimage
from torch.utils.data import DataLoader
from hamcrest import assert_that, instance_of, has_length, all_of, has_property, equal_to, calling, raises

from deepchecks.utils.strings import get_random_string
from deepchecks.vision.datasets.classification import simple_classification_data


PARENT_FOLDER = pathlib.Path(__file__).absolute().parent


@pytest.fixture
def correct_images_folder():
    image = pilimage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))

    images_folder = PARENT_FOLDER / get_random_string()
    images_folder.mkdir()
    (images_folder / "train").mkdir()
    (images_folder / "train" / "class1").mkdir()
    (images_folder / "test").mkdir()
    (images_folder / "test" / "class1").mkdir()

    image.save(str(images_folder / "train" / "class1" / "image.png"))
    image.save(str(images_folder / "test" / "class1" / "image.png"))

    yield images_folder

    rmtree(str(images_folder))


@pytest.fixture
def incorrect_images_folder():
    image = pilimage.fromarray(np.zeros((10, 10, 3), dtype=np.uint8))

    images_folder = PARENT_FOLDER / get_random_string()
    images_folder.mkdir()
    (images_folder / "train").mkdir()
    (images_folder / "test").mkdir()

    image.save(str(images_folder / "train" / "image.png"))
    image.save(str(images_folder / "test" / "image.png"))

    yield images_folder

    rmtree(str(images_folder))


def test_load_simple_classification_dataset(correct_images_folder):
    dataloader = simple.load_dataset(
        root=str(correct_images_folder),
        object_type="DataLoader",
        image_extension="PNG"
    )

    assert_that(dataloader, all_of(
        instance_of(DataLoader),
        has_length(1),
        has_property("dataset", instance_of(simple.SimpleClassificationDataset))
    ))

    img, label = dataloader.dataset[0]

    assert_that(img, all_of(
        instance_of(np.ndarray),
        has_property("shape", equal_to((10, 10, 3)))
    ))

    assert_that(label, instance_of(int))
    assert_that(dataloader.dataset.reverse_classes_map[label] == "class1")


def test_load_simple_classification_vision_data(correct_images_folder):
    vision_data = simple.load_dataset(
        root=str(correct_images_folder),
        object_type="VisionData",
        image_extension="PNG"
    )

    batches = list(vision_data)
    assert_that(len(batches) == 1)

    images = vision_data.batch_to_images(batches[0])
    labels = vision_data.batch_to_labels(batches[0])
    assert_that(len(images) == 1 and len(labels) == 1)

    assert_that(images[0], all_of(
        instance_of(np.ndarray),
        has_property("shape", equal_to((10, 10, 3)))
    ))
    assert_that(vision_data._data_loader.dataset.reverse_classes_map[labels[0].item()] == "class1")


def test_load_simple_classification_dataset_from_broken_folder(incorrect_images_folder):
    assert_that(
        calling(simple.load_dataset).with_args(
            root=str(incorrect_images_folder),
            object_type="DataLoader",
            image_extension="PNG"),
        raises(ValueError)
    )