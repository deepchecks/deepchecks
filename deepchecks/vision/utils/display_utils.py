# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------
#
"""Module for displaying images and labels."""
import typing as t

import matplotlib.pyplot as plt
import numpy as np

from deepchecks.vision.utils.image_functions import draw_bboxes, prepare_thumbnail
from deepchecks.vision.vision_data import TaskType


def visualize_vision_data(dataset, n_show: int = 6):
    """Display images and labels from the dataset.

    Parameters:
    -----------
    dataset: VisionData
        The dataset to display.
    n_show: int, default: 6
        Number of images to display.
    """
    sampled_ds = dataset.copy(n_show, shuffle=False, random_state=0)
    batch = next(iter(sampled_ds))
    images = sampled_ds.batch_to_images(batch)
    labels = sampled_ds.batch_to_labels(batch)

    _, m_axs = plt.subplots(2, int(n_show / 2.), figsize=(10 * (n_show / 6.), 6))
    m_axs = m_axs.flatten()
    for i in range(len(images)):
        m_axs[i].imshow(images[i])
        m_axs[i].set_title(sampled_ds.label_id_to_name(int(labels[i])))
        m_axs[i].axis('off')
    plt.show()


def draw_image(image: np.ndarray, label, task_type: TaskType,
               thumbnail_size: t.Tuple[int, int] = (200, 200), draw_label: bool = True) -> str:
    """Return an image to show as output of the display.

    Parameters
    ----------
    image : np.ndarray
        The image to draw, must be a [H, W, C] 3D numpy array.
    label :
        2-dim labels tensor for the image to draw on top of the image, shape depends on task type.
    task_type : TaskType
        The task type associated with the label.
    thumbnail_size: t.Tuple[int,int]
        The required size of the image for display.
    draw_label : bool, default: True
        Whether to draw the label on the image or not.
    Returns
    -------
    str
        The image in the provided thumbnail size with the label drawn on top of it for relevant tasks as html.
    """
    if draw_label and task_type == TaskType.OBJECT_DETECTION:
        image = draw_bboxes(image, label, copy_image=False, border_width=5)
    image_thumbnail = prepare_thumbnail(
        image=image,
        size=thumbnail_size,
        copy_image=False
    )
    return image_thumbnail
