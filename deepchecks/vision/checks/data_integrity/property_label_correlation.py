from collections import defaultdict
from typing import Any, Dict, Hashable, List, Optional, TypeVar, Union

from deepchecks.core import DatasetKind
from deepchecks.vision import Context, SingleDatasetCheck, Batch
from deepchecks.vision.task_type import TaskType
from deepchecks.vision.utils.image_properties import default_image_properties
from deepchecks.vision.utils.vision_properties import PropertiesInputType
from deepchecks.vision.utils.image_functions import crop_image
from deepchecks.vision.utils.property_label_correlation_utils import calc_properties_for_property_label_correlation

__all__ = ['PropertyLabelCorrelation']
FLC = TypeVar('FLC', bound='PropertyLabelCorrelation')


class PropertyLabelCorrelation(SingleDatasetCheck):
    """
    Return the Predictive Power Score of image properties, in order to estimate their ability to predict the label.

    The PPS represents the ability of a feature to single-handedly predict another feature or label.
    In this check, we specifically use it to assess the ability to predict the label by an image property (e.g.
    brightness, contrast etc.)
    A high PPS (close to 1) can mean that there's a bias in the dataset, as a single property can predict the label
    successfully, using simple classic ML algorithms - meaning that a deep learning algorithm may accidentally learn
    these properties instead of more accurate complex abstractions.
    For example, in a classification dataset of wolves and dogs photographs, if only wolves are photographed in the
    snow, the brightness of the image may be used to predict the label "wolf" easily. In this case, a model might not
    learn to discern wolf from dog by the animal's characteristics, but by using the background color.

    For classification tasks, this check uses PPS to predict the class by image properties.
    For object detection tasks, this check uses PPS to predict the class of each bounding box, by the image properties
    of that specific bounding box.

    Uses the ppscore package - for more info, see https://github.com/8080labs/ppscore

    Parameters
    ----------
    image_properties : List[Dict[str, Any]], default: None
        List of properties. Replaces the default deepchecks properties.
        Each property is dictionary with keys 'name' (str), 'method' (Callable) and 'output_type' (str),
        representing attributes of said method. 'output_type' must be one of:
        - 'numeric' - for continuous ordinal outputs.
        - 'categorical' - for discrete, non-ordinal outputs. These can still be numbers,
          but these numbers do not have inherent value.
        For more on image / label properties, see the :ref:`property guide </user-guide/vision/vision_properties.rst>`
    per_class : bool, default: True
        boolean that indicates whether the results of this check should be calculated for all classes or per class in
        label. If True, the conditions will be run per class as well.
    n_top_properties: int, default: 5
        Number of features to show, sorted by the magnitude of difference in PPS
    random_state: int, default: None
        Random state for the ppscore.predictors function
    min_pps_to_show: float, default 0.05
            Minimum PPS to show a class in the graph
    ppscore_params: dict, default: None
        dictionary of additional parameters for the ppscore predictor function
    """
    def __init__(
            self,
            image_properties: Optional[List[Dict[str, Any]]] = None,
            n_top_properties: int = 3,
            per_class: bool = True,
            random_state: int = None,
            min_pps_to_show: float = 0.05,
            ppscore_params: dict = None,
            **kwargs
    ):
        super().__init__(**kwargs)

        self.image_properties = image_properties if image_properties else default_image_properties

        self.min_pps_to_show = min_pps_to_show
        self.per_class = per_class
        self.n_top_properties = n_top_properties
        self.random_state = random_state
        self.ppscore_params = ppscore_params or {}
        self._properties_results = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind: DatasetKind):
        """Calculate image properties for train or test batches."""
        data_for_properties, target = calc_properties_for_property_label_correlation(
            context, batch, dataset_kind, self.image_properties)

        self._properties_results['target'] += target

        for prop_name, property_values in data_for_properties.items():
            self._properties_results[prop_name].extend(property_values)

