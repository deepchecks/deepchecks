.. formatters:

.. currentmodule:: deepchecks.vision.utils

=============================================
The Image, Labels, and Prediction Formatters
=============================================

The formatter objects in the deepchecks.vision module were designed to enable access to the data while making the formatting easy and maintainable.
They are responsible for converting the data (whether these are images, labels, or predictions) to a format deepchecks can work with.

For checks that need the image's data, see :ref:`Image Formatters <formatters__image_formatter>`.
For prediction and label formats for **classification tasks**, see :ref:`Classification Formatters <formatters__classification_formatter>`.
For prediction and label formats for **object detection** tasks, see :ref:`Object Detection Formatters <formatters__object_detection_formatter>`.


Why is This Important?
========================
There is no standard when it comes to data formats in computer vision. Each model can have its own supported data format for the output predictions, and each new benchmark dataset can propose a new labeling format. Even the images formats can be different from each other (PIL, OpenCV, etc.).

In order to run a model and data-agnostic comprehensive testing suites, the data should be in an accepted format the checks can work with.
The formatter objects provide a structured and repeatable way to do that, and are an important part of the vision module.


Types of Formatters
==========================
Deepchecks requires three data formatters:

- **Image Formatter** - responsible for formatting the input images to the accepted format
- **Prediction Formatter** - responsible for formatting the output of the model to an accepted format (depends on the model type)
- **Label Formatter** - responsible for formatting the targets to an accepted format (depends on the task type)

In order to initialize a new formatter, a transformation function between the input format and the accepted format is required.

.. _formatters__image_formatter:

The Image Formatter
=====================
The image formatter is responsible for transforming the image data that is received as an output from the data loader to the required format for the check's display (e.g. unnormalize the image).
In order to create an image formatter, you need to initialize an instance of the :class:`image_formatters.ImageFormatter` class,
passing the transformation function.

Formats for Task Types
=====================
Each CV task type (e.g. classification, object detection, instance segmentation, etc.) has its own data formats.
For example, in classification tasks a common format for the label is simply the label_id, and for the prediction is a probability vector with the probability for each class. In object detection, there are various formats like the COCO format, PASCAL VOC XML format, YOLO format, and more...

Deepchecks accepts a specific data format for each task type and for some types, provides utility functions to convert between common formats to the accepted one.

.. _formatters__classification_formatter:

Classification Format
-------------------------

Accepted format
~~~~~~~~~~~~~~~~~~~~~
For classification tasks, it is required to implement the following formatters:

- **Label**  - should implement an instance of :class:`classification_formatters.ClassificationLabelFormatter`.

- **Predictions** - should implement an instance of :class:`classification_formatters.ClassificationPredictionFormatter`.

.. _formatters__object_detection_formatter:

Object Detection Format
-------------------------

Accepted Format
~~~~~~~~~~~~~~~~
For object detection tasks, it is required to implement the following formatters:

- **Label** - should implement an instance of :class:`detection_formatters.DetectionLabelFormatter`.

- **Detections Predictions** - should implement an instance of :class:`detection_formatters.DetectionPredictionFormatter`.

Converting From Common Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
TBD (Not implemented yet)
