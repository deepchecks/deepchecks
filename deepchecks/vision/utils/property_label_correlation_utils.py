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
"""Module for property label correlation utils."""
from typing import List

from deepchecks.core.errors import ModelValidationError
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType
from deepchecks.vision.vision_data.batch_wrapper import BatchWrapper


def calc_properties_for_property_label_correlation(task_type: TaskType, batch: BatchWrapper, image_properties: List):
    """
    Transform the data to the relevant format and calculate the properties on it.

    Intended for the checks PropertyLabelCorrelation and PropertyLabelCorrelationChange.
    """
    targets = []
    if task_type == TaskType.OBJECT_DETECTION:
        for bboxes_per_image in batch.numpy_labels:
            if bboxes_per_image is not None and len(bboxes_per_image.shape) == 2:
                targets = targets + bboxes_per_image[:, 0].tolist()
        property_type = PropertiesInputType.PARTIAL_IMAGES
    elif task_type == TaskType.CLASSIFICATION:
        targets = targets + batch.numpy_labels
        property_type = PropertiesInputType.IMAGES
    else:
        raise ModelValidationError(f'Check is irrelevant for task of type {task_type}')

    data_for_properties = batch.vision_properties(image_properties, property_type)
    return data_for_properties, targets
