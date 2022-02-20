# -*- coding: utf-8 -*-
"""
=======================
Quickstart in 5 minutes
=======================

In order to run deepchecks.vision, you need two simple things. First,
you’ll need you data and model. Second, you may need to write short
formatter functions that will let us know how to translate your data and
labels into common formats the package can understand. For model and
data, you will need:

-  Your train and test data (each a pytorch dataloader)
-  (optional) A model object, for which calling ``model(batch)`` for a
   dataloder batch returns the batch predictions. Required for running
   checks that need the model’s predictions for running.

To run your first suite on your data and model, you need only a few
lines of code, that start here: `Define a VisionData
Object <#Define-a-VisionData-Object>`__.


.. code:: ipython3

    # If you don't have deepchecks installed yet:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet #--user

"""

from deepchecks.vision.datasets.detection import coco
from deepchecks.vision import VisionData
from deepchecks.vision.utils import DetectionLabelFormatter, DetectionPredictionFormatter
from deepchecks.vision.suites import full_suite
from deepchecks.vision.checks import TrainTestLabelDrift

#%%
# Load Data and Model
# -------------------
#
# For the purpose of this guide we’ll use the `coco 128
# dataset <https://www.kaggle.com/ultralytics/coco128>`__ and the ultralytics
# `yolov5s <https://pytorch.org/hub/ultralytics_yolov5/#load-from-pytorch-hub>`__
# object detection model, both already included in the deepchecks package:

yolo = coco.load_model(pretrained=True)
coco_train_loader = coco.load_dataset(train=True)
coco_test_loader = coco.load_dataset(train=False)

#%%
# Define a VisionData Object
# --------------------------
#
# Once you have your model and data, the next step is wrapping each
# dataloader inside a VisionData object. These objects serve to the bundle
# together several objects: 1. Your dataloader. 1. label_transformer -
# Formatter used to state the task type (Classification, Object Detection
# etc.) and the function that converts the batch labels to a standard
# format. Visit the `User
# Guide <../../../user-guide/vision/formatter_objects.rst>`__, the API
# reference of the
# `classification
# <../../../api/generated/deepchecks.vision.utils.classification_formatters.ClassificationLabelFormatter.rst>`__
# formatters, or the API reference of the `object
# detection <../../../api/generated/deepchecks.vision.utils.detection_formatters.DetectionLabelFormatter.rst>`__
# formatters for more info. 1. image_transformer- Formatter used to
# convert the batch images to images in a standard format. See the `User
# Guide <../../../user-guide/vision/formatter_objects.rst#the-image-formatter>`__
# for more info.
#
# In the following example, no conversion is needed so
# `DetectionLabelFormatter
# <../../../api/generated/deepchecks.vision.utils.detection_formatters.DetectionLabelFormatter.rst>`__
# is used with the unit callable to declare that this is an Object
# Detection dataset.


# The num_classes is explicitly declared to aid computation, but would be inferred automatically otherwise.
train_ds = VisionData(coco_train_loader, label_transformer=DetectionLabelFormatter(coco.yolo_label_formatter),
                      num_classes=80)
test_ds = VisionData(coco_test_loader, label_transformer=DetectionLabelFormatter(coco.yolo_label_formatter),
                     num_classes=80)

#%%
# Run a Deepchecks Suite
# ----------------------
#
# Run the full suite
# ~~~~~~~~~~~~~~~~~~
#
# Use the ``full_suite`` that is a collection of (most of) the prebuilt
# checks. Check out the `when should you use deepchecks
# guide <../../../user-guide/concepts/when_should_you_use.rst>`__ for some
# more info about the existing suites and when to use them.


suite = full_suite()

#%%
# The `ClassPerformance <../checks/performance/class_performance.ipynb>`__
# check is one of the checks included in this suite, and this check
# computes Object Detection metrics using the labels and the predictions
# of the yolo model.
#
# In order to function, we must pass the check (and thus the suite) a
# prediction_formatter function that converts the yolo predictions to a
# standard format. In this example, the formatter is already included in
# the package. For more info about prediction formatters, visit the `User
# Guide <../../../user-guide/vision/formatter_objects.rst>`__, the API
# reference of the
# `classification
# <../../../api/generated/deepchecks.vision.utils.classification_formatters.ClassificationPredictionFormatter.rst>`__
# formatters, or the API reference of the `object
# detection <../../../api/generated/deepchecks.vision.utils.detection_formatters.DetectionPredictionFormatter.rst>`__
# formatters.

det_formatter = DetectionPredictionFormatter(coco.yolo_prediction_formatter)
result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=yolo, prediction_formatter=det_formatter)

#%%
# In order to view the results, the result object can be exported to an
# html file, as demonstrated here.

result.save_as_html('full_suite_result.html')

#%%
# If the code is running inside a jupyter notebook, the result can also be
# viewed by simply running ``result`` inside a notebook cell, as
# demonstrated here.

result

#%%
# Run a Deepchecks Check
# ----------------------
#
# If you want to run a specific check, you can just import it and run it
# directly.
#
# Check out the `Check Demonstrations <../index.rst>`__ in the examples or
# the `API Reference <../../../api/index.rst>`__ for more info about the
# existing checks and their parameters.

check = TrainTestLabelDrift()
result = check.run(train_ds, test_ds)
result

#%%
# and also inspect the result value which has a check-dependant structure:

print(result.value)