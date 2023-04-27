# -*- coding: utf-8 -*-
"""

.. _vision__weak_segments_performance:

Weak Segments Performance
*************************

This notebook provides an overview for using and understanding the weak segment performance check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Automatically detecting weak segments <#automatically-detecting-weak-segments>`__
* `Generate Dataset <#generate-dataset>`__
* `Run the check <#run-the-check>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
==================================

The check is designed to easily identify the model's weakest segments.
The segments are characterized by the :ref:`image properties <vision__properties_guide>` such as
contrast and aspect ratio.

Automatically detecting weak segments
=====================================

The check performs several steps:

#. We calculate the image properties for each sample. The properties to calculate can be passed explicitly or resort to
   the default image properties.

#. We calculate loss for each sample in the dataset using the provided model or predictions, the loss function can be
   passed explicitly or set to a default based on the task type.

#. We train multiple simple tree based models, each one is trained using two
   properties to predict the per sample error calculated before.

#. We extract the corresponding data samples for each of the leaves in each of the trees (data segments) and calculate
   the model performance on them. For the weakest data segments detected we also calculate the model's
   performance on data segments surrounding them.
"""

#%%
# Generate Dataset
# =================
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection import coco_tensorflow as coco

from deepchecks.vision.checks import WeakSegmentsPerformance
from deepchecks.vision.datasets.detection import coco_torch as coco

coco_data = coco.load_dataset(train=False, object_type='VisionData')

#%%
# Run the check
# =============
check = WeakSegmentsPerformance()
result = check.run(coco_data)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()

#%%
# The result will be displayed in a new window.

#%%
# Observe the check's output
# --------------------------
#
# We see in the results that the check indeed found several segments on which the model performance is below average.
# In the heatmap display we can see the model's performance on the weakest segments and their environment with respect
# to the two segmentation features. In order to get the full list of weak segments found we can look at
# the ``result.value`` attribute. Shown below are the 3 segments with the worst performance.


result.value['weak_segments_list'].head(3)

#%%
# Now we will run a check with properties and minimum segment size ratio (the minimal fraction of the data to be
# considered as a segment) different from the defaults.
from deepchecks.vision.utils.image_properties import brightness, texture_level
properties = [{'name': 'brightness', 'method': brightness, 'output_type': 'numerical'},
              {'name': ' texture', 'method': texture_level, 'output_type': 'numerical'}]
check = WeakSegmentsPerformance(segment_minimum_size_ratio=0.03, image_properties=properties)
result = check.run(coco_data)
result.show()


#%%
# Define a condition
# ==================
#
# We can add a condition that will validate the model's performance on the weakest segment detected is above a certain
# threshold. A scenario where this can be useful is when we want to make sure that the model is not under performing
# on a subset of the data that is of interest to us.

# Let's add a condition and re-run the check:

check.add_condition_segments_relative_performance_greater_than(0.1)
result = check.run(coco_data)
result.show(show_additional_outputs=False)
