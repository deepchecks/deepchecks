# -*- coding: utf-8 -*-
"""
Test Your Vision Data Class During Development
================================================
"Data Classes" are used to transform the structure of your data to the
structure required for deepchecks. To help ensure they work
as intended, deepchecks has built-in helper functions for
validating their structure. This guide will demonstrate how to use this helper function
to implement them for your own data, step by step.


**Structure:**

* `Load data and model <#load-data-and-model>`__
* `Create simple DetectionData object <#create-simple-detectiondata-object>`__
* `Running the extractors validation <#running-the-extractors-validation>`__
* `Understand validation results <#understand-validation-results>`__
* `The end result <#the-end-result>`__
"""
#%%
# Load data and model
# -------------------
# In the first step we load the DataLoader and our model

from deepchecks.vision.datasets.detection.coco import load_dataset, load_model

data_loader = load_dataset(train=False, batch_size=1000, object_type='DataLoader')
model = load_model()

#%%
# Create simple DetectionData object
# ----------------------------------
# In the second step since this is an object detection task we will override a
# DetectionData object with simple extracting functions. We know our DataLoader
# and model's output are not in the format expected by deepchecks, so when we
# validate them on our data we will see in the results the functions we overrided
# are not passing, and then we will implement a correct functions.

from deepchecks.vision.detection_data import DetectionData

class CocoDetectionData(DetectionData):
    def batch_to_images(self, batch):
        return batch[0]

    def batch_to_labels(self, batch):
        return batch[1]

    def infer_on_batch(self, batch, model, device):
        return model.to(device)(batch[0])

#%%
# Running the extractors validation
# ---------------------------------
# Now we will load our validate function and see the results while running
# the extractors on our data. The function will print for us the validation
# results. At the end, if all your extractors are valid the output should look
# in `this <#The-end-result>`__.

from deepchecks.vision.utils.validation import validate_extractors

validate_extractors(CocoDetectionData(data_loader), model)

#%%
# Understand validation results
# -----------------------------
# When looking at the result first thing we see is that it's separated into 2 parts.
#
# First one is about the structure we expect to get. This validation is automatic
# since it's purely technical and doesn't check content correctness. For example,
# in our validation above we see that the label extractor is passing, meaning the
# labels are returning in the expected format. Second part is about the content,
# which can't be automatically validated and requires your attention. This part
# includes looking visually at data outputted by the formatters to validate it is
# right. In the validation above we see a list of classes that doesn't seem to make
# sense. This is because although our labels are in the right structure, the content
# inside is not valid.
#
# We know that the classes in our data are represented by class id which is an int,
# therefore we understand the labels does not contain the data in the right order.
# For the next step we'll fix the label extractor and then validate again:

import torch

class CocoDetectionData(DetectionData):
    def batch_to_labels(self, batch):
        # Translate labels to deepchecks format.
        # the label_id here is in the last position of the tensor, and the DetectionLabelFormatter expects it
        # at the first position.
        formatted_labels = []
        for tensor in batch[1]:
            tensor = torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3])) if len(tensor) > 0 else tensor
            formatted_labels.append(tensor)
        return formatted_labels

    def batch_to_images(self, batch):
        return batch[0]

    def infer_on_batch(self, batch, model, device):
        return model.to(device)(batch[0])

validate_extractors(CocoDetectionData(data_loader), model)

#%%
#  Now we can see in the content section that our classes are indeed as we expect
# them to be, class ids of type int. Now we can continue and fix the prediction extractor

class CocoDetectionData(DetectionData):
    def infer_on_batch(self, batch, model, device):
        # Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6]"""
        return_list = []
        predictions = model.to(device)(batch[0])

        # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        for single_image_tensor in predictions.pred:
            return_list.append(single_image_tensor)

        return return_list

    # using the same label extractor
    def batch_to_labels(self, batch):
        # Translate labels to deepchecks format.
        # the label_id here is in the last position of the tensor, and the DetectionLabelFormatter expects it
        # at the first position.
        formatted_labels = []
        for tensor in batch[1]:
            tensor = torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3])) if len(tensor) > 0 else tensor
            formatted_labels.append(tensor)
        return formatted_labels
    
    def batch_to_images(self, batch):
        return batch[0]

validate_extractors(CocoDetectionData(data_loader), model)

#%%
# Now our prediction formatter also have valid structure. But in order to really
# validate it we also need visual assertion and for that we need the image extractor to work.

import numpy as np

class CocoDetectionData(DetectionData):
    def batch_to_images(self, batch):
        # Yolo works on PIL and ImageFormatter expects images as numpy arrays
        return [np.array(x) for x in batch[0]]

    # using the same prediction extractor
    def infer_on_batch(self, batch, model, device):
        # Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6]"""
        return_list = []
        predictions = model.to(device)(batch[0])

        # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        for single_image_tensor in predictions.pred:
            return_list.append(single_image_tensor)

        return return_list

    # using the same label extractor
    def batch_to_labels(self, batch):
        # Translate labels to deepchecks format.
        # the label_id here is in the last position of the tensor, and the DetectionLabelFormatter expects it
        # at the first position.
        formatted_labels = []
        for tensor in batch[1]:
            tensor = torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3])) if len(tensor) > 0 else tensor
            formatted_labels.append(tensor)
        return formatted_labels


validate_extractors(CocoDetectionData(data_loader), model)

#%%
# Now that that image extractor is valid it displays for us visually the label and prediction.
# When we look at the label we see it is correct, but when we look at the prediction something
# is broken.
#
# We need to fix the prediction so the prediction will be returned in
# [x, y, w, h, confidence, class] format.

class CocoDetectionData(DetectionData):
    def infer_on_batch(self, batch, model, device):
        # Convert from yolo Detections object to List (per image) of Tensors of the shape [N, 6] with each row being 
        # [x, y, w, h, confidence, class] for each bbox in the image."""
        return_list = []
        predictions = model.to(device)(batch[0])

        # yolo Detections objects have List[torch.Tensor] xyxy output in .pred
        for single_image_tensor in predictions.pred:
            pred_modified = torch.clone(single_image_tensor)
            pred_modified[:, 2] = pred_modified[:, 2] - pred_modified[:, 0]  # w = x_right - x_left
            pred_modified[:, 3] = pred_modified[:, 3] - pred_modified[:, 1]  # h = y_bottom - y_top
            return_list.append(pred_modified)

        return return_list

    # using the same label extractor
    def batch_to_labels(self, batch):
        # Translate labels to deepchecks format.
        # the label_id here is in the last position of the tensor, and the DetectionLabelFormatter expects it
        # at the first position.
        formatted_labels = []
        for tensor in batch[1]:
            tensor = torch.index_select(tensor, 1, torch.LongTensor([4, 0, 1, 2, 3])) if len(tensor) > 0 else tensor
            formatted_labels.append(tensor)
        return formatted_labels

    # using the same image extractor
    def batch_to_images(self, batch):
        # Yolo works on PIL and ImageFormatter expects images as numpy arrays
        return [np.array(x) for x in batch[0]]

#%%
# The end result
# --------------
validate_extractors(CocoDetectionData(data_loader), model)
