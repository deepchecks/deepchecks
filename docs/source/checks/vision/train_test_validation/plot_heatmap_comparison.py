# -*- coding: utf-8 -*-
"""
.. _plot_vision_heatmap_comparison:

Heatmap Comparison
******************

This notebooks provides an overview for using and understanding Heatmap comparison check.

**Structure:**

* `What Is a Heatmap Comparison? <#what-is-a-heatmap-comparison>`__
* `Run the Check on a Classification Task <#run-the-check-on-a-classification-task-mnist>`__
* `Run the Check on an Object Detection Task <#run-the-check-on-an-object-detection-task-coco>`__
* `Limit to Specific Classes <#limit-to-specific-classes>`__

What Is a Heatmap Comparison?
=============================
Heatmap comparison is a method of detecting data drift in image data. Data drift is
simply a change in the distribution of data over time or between several distinct cases.
It is also one of the top reasons that a machine learning model performance degrades
over time, or when applied to new scenarios.

The **Heatmap comparison** check simply
computes an average image for all images in each dataset, train and test, and visualizes
both the average images of both. That way, we can visually compare the difference between
the datasets' brightness distribution. For example, if training data contains significantly
more images with sky, we will see that the average train image is brighter in the upper half
of the heatmap.

Comparing Labels for Object Detection
-------------------------------------
For object detection tasks, it is also possible to visualize Label Drift, by displaying the average of bounding
box label coverage. This is done by producing label
maps per image, in which each pixel inside a bounding box is white and the rest and black.
Then, the average of all these images is displayed.

In our previous example, the drift caused by more images with sky in training would also
be visible by a lack of labels in the upper half of the average label map of the training
data, due to lack of labels in the sky.

Other Methods of Drift Detection
--------------------------------
Another, more traditional method to detect such drift would be to use statistical methods.
Such an approach is covered by several builtin check in the deepchecks.vision package, such as the
:doc:`Label Drift Check
</checks_gallery/vision/train_test_validation/plot_train_test_label_drift>`
or the :doc:`Image Dataset Drift Check
</checks_gallery/vision/train_test_validation/plot_image_dataset_drift>`.


Run the Check on a Classification Task (MNIST)
==============================================
"""

#%%
# Imports
# -------
#
# .. note::
#   In this example, we use the pytorch version of the mnist dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.classification.mnist_tensorflow import load_dataset

from deepchecks.vision.datasets.classification.mnist_torch import load_dataset

#%%
# Loading Data
# ------------


mnist_data_train = load_dataset(train=True, batch_size=64, object_type='VisionData')
mnist_data_test = load_dataset(train=False, batch_size=64, object_type='VisionData')

#%%

from deepchecks.vision.checks import HeatmapComparison

check = HeatmapComparison()
result = check.run(mnist_data_train, mnist_data_test)
result

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# Run the Check on an Object Detection Task (Coco)
# ================================================
#
# .. note::
#   In this example, we use the pytorch version of the coco dataset and model. In order to run this example using
#   tensorflow, please change the import statements to::
#
#       from deepchecks.vision.datasets.detection.coco_tensorflow import load_dataset

from deepchecks.vision.datasets.detection.coco_torch import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%

check = HeatmapComparison()
result = check.run(train_ds, test_ds)
result

#%%
# Limit to Specific Classes
# =========================
# The check can be limited to compare the bounding box coverage for a specific set
# of classes. We'll use that to inspect only objects labeled as human (class_id 0)

check = HeatmapComparison(classes_to_display=['person'])
result = check.run(train_ds, test_ds)
result

#%%
# We can see a significant increased abundance of humans in the test data, located
# in the images lower center!
