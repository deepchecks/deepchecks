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
"""Module contains LabelPropertyOutliers check."""
import typing as t

from deepchecks.core.errors import DeepchecksProcessError
from deepchecks.vision._shared_docs import docstrings
from deepchecks.vision.checks.data_integrity.abstract_property_outliers import AbstractPropertyOutliers
from deepchecks.vision.utils import label_prediction_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.vision_data import TaskType, VisionData

__all__ = ['LabelPropertyOutliers']


@docstrings
class LabelPropertyOutliers(AbstractPropertyOutliers):
    """Find outliers labels with respect to the given properties.

    The check computes several label properties and then computes the number of outliers for each property.
    The check uses `IQR <https://en.wikipedia.org/wiki/Interquartile_range#Outliers>`_ to detect outliers out of the
    single dimension properties.

    Parameters
    ----------
    label_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is a dictionary with keys ``'name'`` (str), ``method`` (Callable) and ``'output_type'`` (str),
        representing attributes of said method. 'output_type' must be one of:

        - ``'numeric'`` - for continuous ordinal outputs.
        - ``'categorical'`` - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        - ``'class_id'`` - for properties that return the class_id. This is used because these
          properties are later matched with the ``VisionData.label_map``, if one was given.

        For more on image / label properties, see the guide about :ref:`vision_properties_guide`.
    n_show_top : int , default: 3
        number of outliers to show from each direction (upper limit and bottom limit)
    iqr_percentiles: Tuple[int, int], default: (25, 75)
        Two percentiles which define the IQR range
    iqr_scale: float, default: 1.5
        The scale to multiply the IQR range for the outliers detection
    {additional_check_init_params:2*indent}
    """

    def __init__(self, label_properties: t.List[t.Dict[str, t.Any]] = None, n_show_top: int = 3,
                 iqr_percentiles: t.Tuple[int, int] = (25, 75), iqr_scale: float = 1.5,
                 n_samples: t.Optional[int] = 10000, **kwargs):
        super().__init__(properties_list=label_properties, property_input_type=PropertiesInputType.LABELS,
                         n_show_top=n_show_top, iqr_percentiles=iqr_percentiles, iqr_scale=iqr_scale,
                         draw_label_on_image=True, n_samples=n_samples, **kwargs)

    def get_default_properties(self, data: VisionData):
        """Return default properties to run in the check."""
        if data.task_type == TaskType.CLASSIFICATION:
            raise DeepchecksProcessError('task type classification does not have default label '
                                         'properties for label outliers.')
        elif data.task_type == TaskType.OBJECT_DETECTION:
            return label_prediction_properties.DEFAULT_OBJECT_DETECTION_LABEL_PROPERTIES
        elif data.task_type == TaskType.SEMANTIC_SEGMENTATION:
            return label_prediction_properties.DEFAULT_SEMANTIC_SEGMENTATION_LABEL_PROPERTIES
        else:
            raise DeepchecksProcessError(f'task type {data.task_type} does not have default label '
                                         f'properties defined.')
