# -*- coding: utf-8 -*-
"""
Quickstart in 5 minutes
=======================

In order to run deepchecks.vision, you need two simple things. First,
you'll need you data and model. Second, you may need to write short
formatter functions that will let us know how to translate your data and
labels into common formats the package can understand. For model and
data, you will need:

-  Your train and test data (each a pytorch dataloader)
-  (optional) A model object, for which calling ``model(batch)`` for a
   dataloader batch returns the batch predictions. Required for running
   checks that need the model's predictions for running.

To run your first suite on your data and model, you need only a few
lines of code, that start here: `Define a VisionData
Object <#fefine-a-visiondata-object>`__.


.. code:: python

    # If you don't have deepchecks installed yet:
    import sys
    !{sys.executable} -m pip install deepchecks -U --quiet #--user

"""
import warnings
from typing import Union, List

import torch
import numpy as np

from deepchecks.vision.detection_data import DetectionData
from deepchecks.vision.datasets.detection import coco
from deepchecks.vision import VisionData
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

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
yolo = coco.load_model(pretrained=True, device=device)
coco_train_loader = coco.load_dataset(train=True)
coco_test_loader = coco.load_dataset(train=False)

#%%
# Implement a Data Object
# --------------------------
#
# Deepchecks’ checks and suites expect a data object that is specific for the task type. These data objects are used to
# load a preprocess data for the particular task type, and all inherit from
# `Vision Data <../../../api/generated/deepchecks.vision.VisionData.rst>`__.
#
# For classification - the data class should inherit from
# `ClassificationData <../../../api/generated/deepchecks.vision.ClassificationData.rst`__, and for detection from
# `DetectionData <../../../api/generated/deepchecks.vision.DetectionData.rst>`__.
#
# Because our example here is a detection problem, we will create a class inherited from the DetectionData class, and
# will implement the following requried functions:
#
# - batch_to_images - Transform a batch of data to images in the
# accpeted format. For more info refer to the
# `API reference <../../../api/generated/deepchecks.vision.DetectionData.batch_to_images.rst>`__.
# - batch_to_labels - Extract the labels from a batch of data. For more info refer to the
# `API reference <../../../api/generated/deepchecks.vision.DetectionData.batch_to_labels.rst>`__.
# - infer_on_batch - Return the predictions of the model on a batch of data. For more info refer to the
# `API reference <../../../api/generated/deepchecks.vision.DetectionData.infer_on_batch.rst>`__.


class COCOData(DetectionData):

    def batch_to_labels(self, batch) -> Union[List[torch.Tensor], torch.Tensor]:
        def move_class(tensor):
            return torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3]).to(tensor.device)) \
                if len(tensor) > 0 else tensor

        return [move_class(tensor) for tensor in batch[1]]

    def infer_on_batch(self, batch, model, device) -> Union[List[torch.Tensor], torch.Tensor]:
        return_list = []

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=UserWarning)

            predictions: 'ultralytics.models.common.Detections' = model.to(device)(batch[0])  # noqa: F821

            # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
            for single_image_tensor in predictions.pred:
                pred_modified = torch.clone(single_image_tensor)
                pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
                pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top
                return_list.append(pred_modified)

        return return_list

    def batch_to_images(self, batch) -> List[np.ndarray]:
        return [np.array(x) for x in batch[0]]

#%%
# Now, we will initialize instances of our COCOData class.


train_ds = COCOData(coco_train_loader, label_map=coco.LABEL_MAP)
test_ds = COCOData(coco_test_loader, label_map=coco.LABEL_MAP)

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

result = suite.run(train_dataset=train_ds, test_dataset=test_ds, model=yolo, device=device)

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
result = check.run(train_ds, test_ds, device=device)
result

#%%
# and also inspect the result value which has a check-dependant structure:

print(result.value)
