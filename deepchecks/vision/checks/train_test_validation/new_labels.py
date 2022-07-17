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
from secrets import choice
from typing import Dict

import torch

from deepchecks.core import CheckResult, ConditionResult, DatasetKind
from deepchecks.core.checks import ReduceMixin
from deepchecks.core.condition import ConditionCategory
from deepchecks.core.errors import DeepchecksValueError
from deepchecks.utils.strings import format_number, format_percent
from deepchecks.vision import Batch, Context, TrainTestCheck, VisionData
from deepchecks.vision.utils.image_functions import draw_bboxes, prepare_thumbnail
from deepchecks.vision.vision_data import TaskType

__all__ = ['NewLabels']


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


class NewLabels(TrainTestCheck, ReduceMixin):
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

    def update(self, context: Context, batch: Batch, dataset_kind):
        """No additional caching required for this check."""
        pass

    def compute(self, context: Context) -> CheckResult:
        """Calculate which class_id are only available in the test data set and display them.

        Returns
        -------
        CheckResult
            value: A dictionary showing new class_ids introduced in the test set and number of times they were spotted.
            display: Images containing said class_ids from the test set.
        """
        test_data = context.get_data_by_kind(DatasetKind.TEST)

        classes_in_train = context.get_data_by_kind(DatasetKind.TRAIN).classes_indices.keys()
        classes_only_in_test_count = {key: value for key, value in test_data.n_of_samples_per_class.items()
                                      if key not in classes_in_train}
        # sort by number of appearances in test set in descending order
        classes_only_in_test_count = dict(sorted(classes_only_in_test_count.items(), key=lambda item: -item[1]))

        result_value = {
            'new_labels': {test_data.label_id_to_name(key): value for key, value in classes_only_in_test_count.items()},
            'all_labels_count': sum(test_data.n_of_samples_per_class.values())
        }

        if context.with_display:
            # Create display
            displays = []
            for class_id, num_occurrences in classes_only_in_test_count.items():
                # Create id of alphabetic characters
                sid = ''.join([choice(string.ascii_uppercase) for _ in range(3)])
                images_of_class_id = \
                    list(set(test_data.classes_indices[class_id]))[:self.max_images_to_display_per_label]
                images_combine = ''.join([f'<div class="{sid}-item">{draw_image(test_data, x, class_id)}</div>'
                                          for x in images_of_class_id])

                html = HTML_TEMPLATE.format(
                    label_name=test_data.label_id_to_name(class_id),
                    images=images_combine,
                    count=format_number(num_occurrences),
                    id=sid
                )
                displays.append(html)
                if len(displays) == self.max_new_labels_to_display:
                    break
        else:
            displays = None

        return CheckResult(result_value, display=displays)

    def reduce_output(self, check_result: CheckResult) -> Dict[str, float]:
        """Reduce check result value.

        Returns
        -------
        Dict[str, float]
            number of samples per each new label
        """
        return check_result.value['new_labels']

    def add_condition_new_label_ratio_less_or_equal(self, max_allowed_new_labels_ratio: float = 0.005):
        # Default value set to 0.005 because of sampling mechanism
        """
        Add condition - Ratio of labels that appear only in the test set required to be less or equal to the threshold.

        Parameters
        ----------
        max_allowed_new_labels_ratio: float , default: 0.005
            the max threshold for percentage of labels that only apper in the test set.
        """

        def condition(result: Dict) -> ConditionResult:
            total_labels_in_test_set = result['all_labels_count']
            new_labels_in_test_set = sum(result['new_labels'].values())
            percent_new_labels = new_labels_in_test_set / total_labels_in_test_set

            if new_labels_in_test_set > 0:
                top_new_class = list(result['new_labels'].keys())[:3]
                message = f'{format_percent(percent_new_labels)} of labels found in test set were not in train set. '
                message += f'New labels most common in test set: {top_new_class}'
            else:
                message = 'No new labels were found in test set.'

            category = ConditionCategory.PASS if percent_new_labels <= max_allowed_new_labels_ratio else \
                ConditionCategory.FAIL
            return ConditionResult(category, message)

        name = f'Percentage of new labels in the test set is less or equal to ' \
               f'{format_percent(max_allowed_new_labels_ratio)}'
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
