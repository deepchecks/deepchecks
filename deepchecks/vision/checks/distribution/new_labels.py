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
"""Module contains Train Test label Drift check."""
import string
from collections import defaultdict
from secrets import choice
from typing import Dict

import torch

from deepchecks.core import ConditionResult
from deepchecks.core import DatasetKind, CheckResult
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_percent, format_number
from deepchecks.vision import Context, TrainTestCheck, Batch, VisionData
from deepchecks.vision.utils.image_functions import prepare_thumbnail, draw_bboxes

__all__ = ['NewLabels']

from deepchecks.vision.vision_data import TaskType

THUMBNAIL_SIZE = (200, 200)


def draw_image(data: VisionData, sample_index: int, class_id: int) -> str:
    """Return an image to show as output of the display.

    Parameters
    ----------
    data : VisionData
        The vision data object used in the check.
    sample_index : int
        The batch index of the sample to draw the image for.
    class_id : int
        The class_id of the image or relevant bounding box inside image.
    """
    sample = data.data_loader.dataset[sample_index]
    batch = data.to_batch(sample)

    image = data.batch_to_images(batch)[0]
    if data.task_type == TaskType.OBJECT_DETECTION:
        formatted_labels_of_image = data.batch_to_labels(batch)[0].numpy()
        bboxes_of_class_id = torch.tensor([x for x in formatted_labels_of_image if x[0] == class_id])
        image = draw_bboxes(image, bboxes_of_class_id, copy_image=False, border_width=5)

    image_thumbnail = prepare_thumbnail(
        image=image,
        size=THUMBNAIL_SIZE,
        copy_image=False
    )
    return image_thumbnail


class NewLabels(TrainTestCheck):
    """Detects labels that apper only in the test set.

    Parameters
    ----------
    max_images_to_display_per_label : int , default: 3
        maximum number of images to show from each newly found label in the test set.
    max_new_labels_to_display : int , default: 3
        Maximum number of new labels to display in output.
    """

    def __init__(
            self,
            max_images_to_display_per_label: int = 3,
            max_new_labels_to_display: int = 3,
            **kwargs
    ):
        super().__init__(**kwargs)
        # validate input parameters:
        if not isinstance(max_images_to_display_per_label, int):
            raise DeepchecksValueError('max_num_images_to_display_per_label must be an integer')
        if not isinstance(max_new_labels_to_display, int):
            raise DeepchecksValueError('max_num_new_labels_to_display must be an integer')

        self.max_images_to_display_per_label = max_images_to_display_per_label
        self.max_new_labels_to_display = max_new_labels_to_display
        self._class_id_counter = defaultdict(list)

    def update(self, context: Context, batch: Batch, dataset_kind):
        """Count number of appearances for each class_id in train and test."""
        data = context.get_data_by_kind(dataset_kind)
        classes_in_batch = data.get_classes(batch.labels)
        for labels_per_image in classes_in_batch:
            for label in labels_per_image:
                if label not in self._class_id_counter:
                    self._class_id_counter[label] = [0, 0]
                self._class_id_counter[label][dataset_kind == DatasetKind.TEST] += 1

    def compute(self, context: Context) -> CheckResult:
        """Calculate which class_id are only available in the test data set and display them.

        Returns
        -------
        CheckResult
            value: A dictionary showing new class_ids introduced in the test set and number of times they were spotted.
            display: Images containing said class_ids from the test set.
        """
        data = context.get_data_by_kind(DatasetKind.TEST)
        class_id_only_in_test = defaultdict(list)
        for class_id, value in self._class_id_counter.items():
            if value[0] == 0:
                class_id_only_in_test[class_id] = value[1]
        class_id_only_in_test = dict(sorted(class_id_only_in_test.items(), key=lambda item: -item[1]))

        # Create display
        display = []
        for class_id, num_occurrences in class_id_only_in_test.items():
            # Create id of alphabetic characters
            sid = ''.join([choice(string.ascii_uppercase) for _ in range(3)])
            images_of_class_id = list(set(data.classes_indices[class_id]))[:self.max_images_to_display_per_label]
            images_combine = ''.join([f'<div class="{sid}-item">{draw_image(data, x, class_id)}</div>'
                                      for x in images_of_class_id])

            html = HTML_TEMPLATE.format(
                label_name=data.label_id_to_name(class_id),
                images=images_combine,
                count=format_number(num_occurrences),
                id=sid
            )
            display.append(html)
        class_id_only_in_test['all_labels'] = sum(data.n_of_samples_per_class.values())
        return CheckResult(class_id_only_in_test, display=''.join(display[:self.max_new_labels_to_display]))

    def add_condition_new_label_percentage_not_greater_than(self, max_allowed_new_labels: float = 0.005):
        # Default value set to 0.005 because of sampling mechanism
        """
        Add condition - Percentage of labels that apper only in the test set required to be below specified threshold.

        Parameters
        ----------
        max_allowed_new_labels: float , default: 0.005
            the max threshold for percentage of labels that only apper in the test set.
        """

        def condition(result: Dict) -> ConditionResult:
            total_labels_in_test_set = result['all_labels']
            new_labels_in_test_set = sum(result.values()) - total_labels_in_test_set
            present_new_labels = new_labels_in_test_set / total_labels_in_test_set

            if present_new_labels > max_allowed_new_labels:
                massage = f'{format_percent(present_new_labels)} of labels found in test set were not in train set.'
                return ConditionResult(ConditionCategory.FAIL, massage)
            else:
                return ConditionResult(ConditionCategory.PASS)

        name = f'Percentage of new labels in the test set not above {format_percent(max_allowed_new_labels)}.'
        return self.add_condition(name, condition)


HTML_TEMPLATE = """
<style>
    .{id}-container {{
        overflow-x: auto;
        display: flex;
        flex-direction: column;
        gap: 10px;
    }}
    .{id}-row {{
      display: flex;
      flex-direction: row;
      align-items: center;
      gap: 10px;
    }}
    .{id}-item {{
      display: flex;
      min-width: 200px;
      position: relative;
      word-wrap: break-word;
      align-items: center;
      justify-content: center;
    }}
    .{id}-title {{
        font-family: "Open Sans", verdana, arial, sans-serif;
        color: #2a3f5f
    }}
    /* A fix for jupyter widget which doesn't have width defined on HTML widget */
    .widget-html-content {{
        width: -moz-available;          /* WebKit-based browsers will ignore this. */
        width: -webkit-fill-available;  /* Mozilla-based browsers will ignore this. */
        width: fill-available;
    }}
</style>
<h3><b>Label  "{label_name}"</b></h3>
<div>
Appears {count} times in test set.
</div>
<div class="{id}-container">
    <div class="{id}-row">
        {images}
    </div>
</div>
"""
