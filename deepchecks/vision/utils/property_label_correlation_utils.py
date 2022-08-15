from typing import List

import pandas as pd
from pandas.core.dtypes.common import is_float_dtype

from deepchecks import Context
from deepchecks.core import DatasetKind
from deepchecks.vision.batch_wrapper import Batch
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.vision_properties import PropertiesInputType


def calc_properties_for_property_label_correlation(
        context: Context, batch: Batch, dataset_kind: DatasetKind, image_properties: List):
    """
    Transform the data to the relevant format and calculate the properties on it.

    Intended for the checks PropertyLabelCorrelation and PropertyLabelCorrelationChange.
    """

    imgs = []
    target = []

    dataset = context.get_data_by_kind(dataset_kind)

    if dataset.task_type == TaskType.OBJECT_DETECTION:
        for img, labels in zip(batch.images, batch.labels):
            for label in labels:
                label = label.cpu().detach().numpy()
                bbox = label[1:]
                cropped_img = crop_image(img, *bbox)
                if cropped_img.shape[0] == 0 or cropped_img.shape[1] == 0:
                    continue
                class_id = int(label[0])
                imgs += [cropped_img]
                target += [dataset.label_id_to_name(class_id)]
        property_type = PropertiesInputType.BBOXES
    else:
        for img, classes_ids in zip(batch.images, dataset.get_classes(batch.labels)):
            imgs += [img] * len(classes_ids)
            target += list(map(dataset.label_id_to_name, classes_ids))
        property_type = PropertiesInputType.IMAGES

    data_for_properties = batch.vision_properties(imgs, image_properties, property_type)

    return data_for_properties, target


def is_float_column(col: pd.Series) -> bool:
    """Check if a column must be a float - meaning does it contain fractions.

    Parameters
    ----------
    col : pd.Series
        The column to check.

    Returns
    -------
    bool
        True if the column is float, False otherwise.
    """
    if not is_float_dtype(col):
        return False

    return (col.round() != col).any()
