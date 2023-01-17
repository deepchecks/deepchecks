# -*- coding: utf-8 -*-
"""
.. _plot_vision_feature_label_correlation_change:

Property Label Correlation Change
***********************************

This notebook provides an overview for using and understanding the "Property Label Correlation Change" check.

**Structure:**

* `What is the purpose of the check? <#what-is-the-purpose-of-the-check>`__
* `Run check on a Classification task <#run-the-check-on-a-classification-task>`__
* `Run check on an Object Detection task <#run-the-check-on-an-object-detection-task>`__
* `Define a condition <#define-a-condition>`__

What is the purpose of the check?
=================================
The check estimates for every image :doc:`image property </user-guide/vision/vision_properties>`
(such as brightness, contrast etc.) its ability to predict the label by itself. This check can help find:

* A potential bias in one or both datasets, that leads to the labels being
  strongly correlated with simple image properties such as color, brightness,
  aspect ratio and more. This is a critical problem, that will likely stay hidden
  without this check (as it won't pop up when comparing model performance on train
  and test).

The check is based on calculating the predictive power score (PPS) of each image
property. For more details you can read here `how the PPS is calculated
<#how-is-the-predictive-power-score-pps-calculated>`__.

What is a problematic result?
-----------------------------

1. Image properties with a high predictive score - can indicate that there is a
   bias in the dataset, as a single property can predict the label successfully,
   using simple classic ML algorithms.

   This means that a deep learning algorithm may accidentally learn these properties
   instead of more accurate complex abstractions. For example, in a classification
   dataset of wolves and dogs photographs, if only wolves are photographed in the
   snow, the brightness of the image may be used to predict the label "wolf" easily.

   In this case, a model might not learn to discern wolf from dog by the animal's
   characteristics, but by using the background color.
2. A high difference between the PPS scores of a certain image property in the
   train and in the test datasets - this is an indication for a drift between
   the relation of the property and the label and a possible bias in one of
   the datasets.

   For example: an object detection dataset that identifies household items.
   In it, a pen would usually create a long and thin rectangle bounding box.
   If in the test dataset the pens would be angled differently, or other object
   are mistakenly identified as pens, the bounding boxes may have a different
   aspect ratio. In this case, the PPS of the train dataset will be high, while
   the PPS of the test dataset would be low, indicating that a bias in the train
   dataset does not appear in the test dataset, and could indicate the model will
   not be able to infer correctly on the test (or any other) dataset due to drift.

How do we calculate for different vision tasks?
-----------------------------------------------

* For classification tasks, this check uses PPS to predict the class by image properties.
* For object detection tasks, this check uses PPS to predict the class of each
  bounding box, by the image properties of that specific bounding box.
  This means that for each image, this check crops all the sub-images defined by bounding
  boxes, and uses them as inputs as though they were regular classification dataset images.

How is the Predictive Power Score (PPS) calculated?
---------------------------------------------------
The properties' predictive score results in a numeric score between 0 (feature has
no predictive power) and 1 (feature can fully predict the label alone).

The process of calculating the PPS is the following:
"""
#%%
# 1. Extract from the data only the label and the feature being tested
# 2. Drop samples with missing values
# 3. Keep 5000 (this is configurable parameter) samples from the data
# 4. Preprocess categorical columns. For the label using `sklearn.LabelEncoder` and for the feature
#    using `sklearn.OneHotEncoder`
# 5. Partition the data with 4-fold cross-validation
# 6. Train decision tree
# 7. Compare the trained model's performance with naive model's performance as follows:
#
# Regression: The naive model always predicts the median of the label column,
# the metric being used is MAE and the PPS calculation is: :math:`1 - \frac{\text{MAE model}}{\text{MAE naive}}`
#
# Classification: The naive model always predicts the most common class of
# the label column, The metric being used is F1 and the PPS calculation is:
# :math:`\frac{\text{F1 model} - \text{F1 naive}}{1 - \text{F1 naive}}`
#
# .. note::
#
#    All the PPS parameters can be changed by passing to the check the parameter ``ppscore_params``
#
#
# For further information about PPS you can visit the `ppscore github <https://github.com/8080labs/ppscore>`__
# or the following blog post: `RIP correlation. Introducing the Predictive Power Score
# <https://towardsdatascience.com/rip-correlation-introducing-the-predictive-power-score-3d90808b9598>`__

#%%
# Run the check on a Classification task
# ======================================

#%%
# Loading data (MNIST)
# --------------------

import numpy as np
from deepchecks.vision.checks import PropertyLabelCorrelationChange
from deepchecks.vision.datasets.classification.mnist_torch import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%
# Insert bias
# -----------
# Let's see what happens when we insert bias into the dataset.
#
# Specifically, we're going to change the pixel values of the image depending
# on the label (0 to 9) so there is a correlation between brightness of image
# and the label (also a small correlation of the index)

from deepchecks.vision.utils.transformations import un_normalize_batch

def mnist_batch_to_images_with_bias_mod(mod):
    def mnist_batch_to_images_with_bias(batch):
        """Create function which inverse the data normalization."""
        tensor = batch[0]
        tensor = tensor.permute(0, 2, 3, 1)
        ret = un_normalize_batch(tensor, (0.1307,), (0.3081,))
        # add some label/index correlation
        for i, label in enumerate(batch[1]):
            if i % mod != 0:
                ret[i] = np.ones(ret[i].shape) * int(i % 3 + 1) * int(label)

        return ret
    return mnist_batch_to_images_with_bias
#%%

train_ds.batch_to_images = mnist_batch_to_images_with_bias_mod(9)
test_ds.batch_to_images = mnist_batch_to_images_with_bias_mod(2)

#%%
# Run the check
# -------------

check = PropertyLabelCorrelationChange()
result = check.run(train_ds, test_ds)
result.show()

#%%
# To display the results in an IDE like PyCharm, you can use the following code:

#  result.show_in_window()
#%%
# The result will be displayed in a new window.

#%%
# We can see that the check detected the bias we inserted, and that the
# brightness property of the image has a high PPS, meaning it can be used to
# solely predict the label.
#
# Run the check on an Object Detection task
# =========================================

#%%
# Loading data (COCO)
# --------------------
from deepchecks.vision.datasets.detection.coco import load_dataset

train_ds = load_dataset(train=True, object_type='VisionData')
test_ds = load_dataset(train=False, object_type='VisionData')

#%%
# Insert bias
# -----------
# Let's now see what happens when we insert bias into the dataset.
#
# Specifically, we're going to change the pixel values of parts of the image
# (where the bounding boxes are), so there is a correlation between brightness
# of image and the label

# Increase the pixel values of all bounding boxes by the labels value:

def coco_batch_to_images_with_bias_mod(mod):
    def coco_batch_to_images_with_bias(batch):
        import numpy as np
        ret = [np.array(x) for x in batch[0]]
        for i, labels in enumerate(train_ds.batch_to_labels(batch)):
            if i % mod != 0:
                for label in labels:
                    x, y, w, h = np.array(label[1:]).astype(int)
                    ret[i][y:y+h, x:x+w] = ret[i][y:y+h, x:x+w].clip(min=200) * int(label[0])
        return ret
    return coco_batch_to_images_with_bias


train_ds.batch_to_images = coco_batch_to_images_with_bias_mod(12)
test_ds.batch_to_images = coco_batch_to_images_with_bias_mod(2)


#%%
# Run the check
# -------------

check = PropertyLabelCorrelationChange(per_class=False)
result = check.run(train_ds, test_ds)
result.show()

#%%
# We can see that the check detected the bias we inserted, and that the brightness
# property of the image has a high PPS, meaning it can be used to solely predict the label.
#
# Define a condition
# ==================
# We can define on our check a condition that will validate that our pps scores aren't
# too high. The check has 2 possible built-in conditions:
# 
# ``add_condition_feature_pps_difference_not_greater_than`` - Validate that the difference in 
# the PPS between train and test is not larger than defined amount (default 0.2)
#
# ``add_condition_feature_pps_in_train_not_greater_than`` - Validate that the PPS scores on
# train dataset are not exceeding a defined amount (default 0.7)
#
# Let's add the conditions, and re-run the check:

check = PropertyLabelCorrelationChange(per_class=False).add_condition_property_pps_difference_less_than(0.1) \
        .add_condition_property_pps_in_train_less_than()
result = check.run(train_dataset=train_ds, test_dataset=test_ds)
result.show(show_additional_outputs=False)
