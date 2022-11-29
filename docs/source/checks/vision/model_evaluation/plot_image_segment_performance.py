# -*- coding: utf-8 -*-
"""
.. _plot_vision_image_segment_performance:

Image Segment Performance
*************************

The ImageSegmentPerformance check is deprecated, please use WeakSegmentsPerformance instead.

This notebook provides an overview for using and understanding image segment performance check.

**Structure:**

* `Why the image segment performance is important? <#why-segment-performance-is-important>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

Why segment performance is important?
=====================================
The check helps to detect segments of your data that are under-performing based on
the basic properties of the image. For example, by default the check would show how
the performance depends on brightness, area and other such properties. Identifying
your models' weak segments might help to address specific issues and improve the
overall performance of the model.
"""

#%%
# Run the check
# =============

from deepchecks.vision.checks import ImageSegmentPerformance
from deepchecks.vision.datasets.detection import coco

coco_data = coco.load_dataset(train=False, object_type='VisionData')
model = coco.load_model()

result = ImageSegmentPerformance().run(coco_data, model)
result

#%%
# If you have a GPU, you can speed up this check by passing it as an argument to .run() as device=<your GPU>
#
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.


#%%
# Observe the check’s output
# --------------------------
# The check segmented the data by different properties and calculated the metrics for each
# segment. As the value of result we return all the information on the different segments:

print(f'Properties: {result.value.keys()}')
print(f'brightness bins: {result.value["Brightness"]}')

#%%
# Define a condition
# ------------------
# The check has a default condition which can be defined. The condition calculates for
# each property & metric the mean score and then looks at the ratio between the lowest
# segment score and the mean score. If this ratio is less than defined threshold, the
# condition fails.
#
# The purpose of the condition is to catch properties segments that are significantly
# worse than the mean - which might indicate a problem.

check = ImageSegmentPerformance().add_condition_score_from_mean_ratio_greater_than(0.5)
result = check.run(coco_data, model)
result

#%%
# In this case the condition identified under-performing segments in the
# properties: mean_blue_relative_intensity, brightness, aspect_ratio, mean_red_relative_intensity
