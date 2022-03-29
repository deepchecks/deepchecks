import numpy as np

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision import Batch
from deepchecks.vision.checks.distribution.abstract_property_outliers import AbstractPropertyOutliers
from deepchecks.vision.utils import label_prediction_properties
from deepchecks.vision.utils.image_functions import draw_bboxes
from deepchecks.vision.vision_data import TaskType, VisionData

__all__ = ['LabelPropertyOutliers']


class LabelPropertyOutliers(AbstractPropertyOutliers):

    def get_default_properties(self, data: VisionData):
        """Return default properties to run in the check."""
        if data.task_type == TaskType.CLASSIFICATION:
            raise DeepchecksProcessError(f'task type classification does not have default label '
                                         f'properties for label outliers.')
        elif data.task_type == TaskType.OBJECT_DETECTION:
            return label_prediction_properties.DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
        else:
            raise DeepchecksProcessError(f'task type {data.task_type} does not have default label '
                                         f'properties defined.')

    def get_relevant_data(self, batch: Batch):
        """Get the data on which the check calculates outliers for."""
        return batch.labels

    def draw_image(self, data: VisionData, sample_index: int, index_of_value_in_sample: int,
                   num_properties_in_sample: int) -> np.ndarray:
        """Return an image to show as output of the display.

        Parameters
        ----------
        data : VisionData
            The vision data object used in the check.
        sample_index : int
            The batch index of the sample to draw the image for.
        index_of_value_in_sample : int
            Each sample property is list, then this is the index of the outlier in the sample property list.
        num_properties_in_sample
            The number of values in the sample's property list.
        """
        batch = data.batch_of_index(sample_index)
        image = data.batch_to_images(batch)[0]

        if data.task_type == TaskType.OBJECT_DETECTION:
            label = data.batch_to_labels(batch)[0]
            # If we have same number of values for sample as the number of bboxes in label, we assume that the
            # property returns value per bounding box, so we filter only the relevant bounding box
            if num_properties_in_sample > 1 and num_properties_in_sample == len(label):
                label = label[index_of_value_in_sample].unsqueeze(dim=0)
            image = draw_bboxes(image, label, copy_image=False, border_width=5)

        return image
