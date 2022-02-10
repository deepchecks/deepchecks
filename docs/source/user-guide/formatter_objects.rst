.. formatters:

.. currentmodule:: deepchecks.vision.utils

=============================================
The Image, Labels, and Prediction formatters
=============================================

The formatter objects in the deepchecks.vision module were designed to provide easy access to the data while making their implementation easy and maintainable.
They are responsible for converting the data (whether these are images, labels, or predictions) to a format deepchecks can work with.

.. contents:: Table of Contents
    :local:
    :depth: 3

Why is this important?
========================
There is no standard when it comes to data formats in computer vision. Each model can have its own supported data format for the output predictions, and each new benchmark dataset can propose a new labeling format. Even the images formats can be different from each other (PIL, OpenCV, etc.).

In order to run a model and data-agnostic comprehensive testing suites, the data should be in an accepted format the checks can work with.
The formatter objects provide an easy way to do that, and are an important part of the vision module.


Types of formatters
==========================
Deepchecks requires three data formatters:

- :ref:`inferred automatically <_formatters__image_formatter>` - responsible for formatting the input images to the accepted format
- **Prediction Formatter** - responsible for formatting the output of the model to an accepted format (depends on the model type)
- **Label Formatter** - responsible for formatting the targets to an accepted format (depends on the task type)

In order to initialize a new formatter, a transformation function between the input format and the accepted format is required.

.. _formatters__image_formatter:

The Image Formatter
=====================
The image formatter is responsible for formatting the image data outputted from the data loader to the required format for check displays.
In order to create an image formatter, you need to initialize an instance of the :class:`image_formatters.ImageFormatter` class,
passing the transformation function.

Task types formats
=====================
Each CV task type (e.g. classification, object detection, instance segmentation, etc.) has its own data formats.
For example, in classification tasks a common format for the label is simply the label_id, and for the prediction is a probability vector with the probability for each class. In object detection, there are various formats like the COCO format, PASCAL VOC XML format, YOLO format, and more...

Deepchecks accepts a specific data format for each task type and for some types, provides utility functions to convert between common formats to the accepted one.

Classification format
-------------------------

Accepted format
~~~~~~~~~~~~~~~~~~~~~
In order to use deepchecks to validate your classification model, it is required to implement a :class:`classification_formatters.ClassificationLabelFormatter` for formatting the labels, and a
:class:`classification_formatters.ClassificationPredictionFormatter` for formatting the predictions.

For the labels, the accepted format is a tensor of shape (N,), when N is the number of samples in the batch. Each element is an integer representing the class index.

For the predictions, the accepted format is a tensor of shape (N, n_classes), when N is the number of samples in batch. Each element is an array of length n_classes
that represents the probability of each class.

Example
~~~~~~~~
Let's assume our dataloader return the following label: [class_id, image_sha1]. In order to transform the label to the
accepted format, we will use the following function:

>>> def to_accepted_format(input_batch_from_loader):
...     return input_batch_from_loader[:, 0]

Then, we will create the formatter:

>>> from deepchecks.vision.utils.classification_formatters import ClassificationLabelFormatter, ClassificationPredictionFormatter
>>> label_formatter = ClassificationLabelFormatter(to_accepted_format)


Object detection format
-------------------------

Accepted Format
~~~~~~~~~~~~~~~~
For object detection tasks, it is required to implement the :class:`detection_formatters.DetectionLabelFormatter` and the
:class:`detection_formatters.DetectionPredictionFormatter` classes.

The accepted format for the labels is a list of length N containing tensors of shape (M, 5), where N is the number of samples,
M is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
the width and height of the bounding box (in pixels) and class_id is the class id.

For the detection, the accepted format is list of length N containing tensors of shape (B, 6), where N is the number of images,
B is the number of bounding boxes detected in the sample and each bounding box is represented by 6 values:
[x, y, w, h, confidence, class_id]. x and y are the coordinates (in pixels) of the upper left corner of the
bounding box, w and h are the width and height of the bounding box (in pixels), confidence is the confidence of
the model and class_id is the class id.


Converting from common formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TBD

Example
~~~~~~~~
Assuming we're using the yolo format, which defines the bbox format as xyxy (top left and botton right points, normalized),
we can use the following function to convert the labels to the accepted format:

>>> def yolo_to_coco(input_batch_from_loader):
...     return [torch.stack([torch.cat((bbox[1:3], bbox[4:] - bbox[1:3], bbox[0]), dim=0) for bbox in image])
...             for image in input_batch_from_loader]

Then, we can create the formatter:

>>> from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter, DetectionPredictionFormatter
>>> label_formatter = DetectionLabelFormatter(yolo_to_coco)

Or, we can simply use the utility conversion since the yolo format is a common format:

>>> from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter, DetectionPredictionFormatter
>>> label_formatter = DetectionLabelFormatter('cxcywhn')
