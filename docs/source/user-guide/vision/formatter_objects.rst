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
- **Prediction Formatter** - responsible for predicting using the model and formatting the output to an accepted format (depends on the model type)
- **Label Formatter** - responsible for formatting the targets to an accepted format (depends on the task type)

In order to initialize a new formatter, a transformation function between the input format and the accepted format is required.

.. _formatters__image_formatter:

The Image Formatter
=====================
The image formatter is responsible for transforming the image data that is received as an output from the data loader to a format that is displayable for the check's display (e.g. unnormalize the image).
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

To initialize `DetectionLabelFormatter` for common label formats the notion of the bbox format notation can be used:

.. code-block:: python

    import torch
    from deepchecks.vision.utils import DetectionLabelFormatter

    yolo_formatter = DetectionLabelFormatter('lcxcywhn') 
    pascal_formatter = DetectionLabelFormatter('lxyxy')
    coco_formatter = DetectionLabelFormatter()  # This is the default format of 'lxywh'
    albumentations_formatter = DetectionLabelFormatter('lxyxyn')

Based on the outlined bbox format notation deepchecks will transformed the labels to the format required by deepchecks, which is 'lxywh'.

Elements of the bbox format notation:
    * l - label (class index) of the object;
    * xy - top-left or bottom-right corner of the bbox. (First 'xy' element is always interpreted as x-min, y-min, and second as x-max, y-max). Coordinates of the bounding box's corners are represented with respect to the top-left corner of the image which has (x, y) coordinates (0, 0);
    * cxcy - xy measured from the center of the image, meaning the center of the image is (0, 0);
    * wh - width and height of the bbox;
    * n - placed at the beginning or at the end of the notation to indicate whether the coordinates are normalized or not. Normalized coordinates are not measured in pixels, but as rather ratio of x / image-width and y / image-height.

It is important to note that `DetectionLabelFormatter` does not work with labels represented by JSON, XML, or python dictionaries.
The label batch processed by `DetectionLabelFormatter` must be an iterable of numpy arrays or pytorch tensors.

Example of using the notation format to get a DetectionLabelFormatter converting from yolo format to deepchecks format:

.. code-block:: python

    import numpy as np
    import torch
    from deepchecks.vision.utils import DetectionLabelFormatter

    yolo_formatter = DetectionLabelFormatter('lcxcywhn') 

    # the only correct way to structure data 
    # that should be passed to the `DetectionLabelFormatter`
    data = (
        [ # images
            np.zeros((200, 200)),  # image number 1
            np.zeros((200, 200)),  # image number 2
        ],
        [ # bboxes
            torch.stack([ # image number 1 bboxes
                torch.tensor([0, 0.20, 0.20, 10, 10]), # bbox number 1
                torch.tensor([1, 0.50, 0.50, 25, 25]) # bbox number 2
            ]),
            torch.stack([ # image number 2 bboxes
                torch.tensor([2, 0.65, 0.70, 10, 10]), # bbox number 1
            ])
        ]
    )

    result = yolo_formatter(data)
    # [
    #   tensor([
    #       [0.0000, 35.0000, 35.0000, 10.0000, 10.0000],
    #       [1.0000, 87.5000, 87.5000, 25.0000, 25.0000]
    #   ]),
    #   tensor([
    #       [2., 125., 135.,  10.,  10.]
    #   ])
    # ]
    # note: 
    # - DetectionLabelFormatter returns only transformed labels
    # - bboxes were transformed to the format lxywh (label, x-min, y-min, width, height)


Validating The Correctness of Your Formatters
=============================================
While writing formatters for your data to fit deepchecks, you need a way to test your work is correct.
For this purpose we have an helper function which runs your formatter and prints their outputs, which you can inspect to make sure they work correctly.
Use this function like so:

.. code-block:: python

    from deepchecks.vision.utils.validation import validate_formatters    
    validate_formatters(data_loader, model, label_formatter, image_formatter, pred_formatter)


For more info about the formatters validating see the guide :doc:`formatters_validation </examples/vision/guides/formatters_validating>`