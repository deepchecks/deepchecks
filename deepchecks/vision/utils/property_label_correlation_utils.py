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

from deepchecks.core import DatasetKind
from deepchecks.tabular import Context
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.vision_properties import PropertiesInputType


def calc_properties_for_property_label_correlation(
        context: Context, batch: Batch, dataset_kind: DatasetKind, image_properties: List):
    """
    Transform the data to the relevant format and calculate the properties on it.

    Intended for the checks PropertyLabelCorrelation and PropertyLabelCorrelationChange.
    """
    dataset = context.get_data_by_kind(dataset_kind)

    target = []

    if dataset.task_type == TaskType.OBJECT_DETECTION:
        for labels in batch.labels:
            for label in labels:
                label = label.cpu().detach().numpy()
                bbox = label[1:]
                # make sure image is not out of bounds
                if round(bbox[2]) + min(round(bbox[0]), 0) <= 0 or round(bbox[3]) <= 0 + min(round(bbox[1]), 0):
                    continue
                class_id = int(label[0])
                target.append(dataset.label_id_to_name(class_id))
        property_type = PropertiesInputType.PARTIAL_IMAGES
    else:
        for classes_ids in dataset.get_classes(batch.labels):
            if len(classes_ids) == 0:
                target.append(None)
            else:
                target.append(dataset.label_id_to_name(classes_ids[0]))
        property_type = PropertiesInputType.IMAGES

    data_for_properties = batch.vision_properties(image_properties, property_type)

    return data_for_properties, target
