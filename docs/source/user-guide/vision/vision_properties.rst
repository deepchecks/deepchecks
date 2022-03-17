===============
Data Properties
===============

Some of the Deepchecks' checks (e.g. train-test drift) use "Properties". Properties are simple one-dimension values that
are extracted either from the images, labels or predictions of the data.

For example, one image property is **brightness**, and one label property is **bounding box area** (for detection
tasks). Each property produces a single value per image or per individual label in the image.

What Are Properties Used For?
=============================

The properties values' distribution is a characteristic of your data that might have hidden a relation to your
model performance. They might come handy in a few situations:

1. New unlabeled data: To approximate how your model will work on the new data, since you don't have labels
   in order to do a performance check. If there is a significant drift between the new data's properties and the
   train's properties, then it might indicate the model's predictions will be less accurate.
2. Low test performance: If your model is having low performance on your test data, a good step in debugging
   it is to check the drift of different properties between the train and the test. If there are some
   properties with a high drift it might help you pinpoint to a relevant causes of the model's low performance.
3. Find bias in the data: Sometimes the model training might be affected the properties we are not aware of,
   and that aren't the core objective we are aiming to learn. For example, in a classification dataset of wolves
   and dogs photographs, if only wolves are photographed in the snow, the brightness of the image may be used to
   predict the label "wolf" easily. In this case, a model might not learn to discern wolf from dog by the animal's
   characteristics, but by using the background color.
4. Find weak segments: The properties can be used to segment the data and test for low performing segments.
   If found, the weak segment might indicate a gap in the training data where the data quality is worse.

Deepchecks' Default Properties
==============================

We divide the properties into 3 groups: image, label and prediction properties.
When running checks you can either create your own properties or rely on Deepchecks' default properties.

The default image properties are:

- Aspect Ratio (height / width)
- Area
- Brightness
- RMS (Root Mean Square) Contrast
- Normalized RGB Mean channels: Mean color intensity for each channel. The color intensity is normalized according to
  the other color channels per pixel. This is done in order to capture the relationships between channels and not just
  general intensity (brightness).

  - Normalized Red Mean
  - Normalized Blue Mean
  - Normalized Green Mean

The default label & predictions properties are:

- Samples Per Class (classification + object detection)
- Bounding Box Area (object detection)
- Number of Bounding Boxes Per Image (object detection)

Property Structure
==================

The structure of all properties is similar, it consists of dictionary with 3 keys:

- `name` - The name of the property
- `method` - A callable function which accepts data (see elaboration bellow), and returns a list of
  values.
- `output_type` - One of the following options:

  - `continuous` - For numeric values with continuous nature
  - `discrete` - For numeric values with discrete nature or non-numeric values
  - `class_id` - Means the output is of class ids. In this case we will try to translate the ids into their
    corresponding class labels.

Each dictionary is a single property, and the checks accepts a list of those dictionaries. For example:

.. code-block:: python

  def mean_image(images):
    return [image.mean() for image in images]

  properties = [
    {'name': 'My Image Mean', 'method': mean_image, 'output_type': 'continuous'}
  ]


The Method's Input
~~~~~~~~~~~~~~~~~~~~~
The 3 types of properties receive different inputs and they are not interchangeable. For each check you will have
to pass the appropriate properties.

The method's input is either images, labels or predictions in
:doc:`Deepchecks' format </user-guide/vision/formatter_objects>`. We will demonstrate the 3 drift checks (for
each property type) and implement the properties to pass to it.

Image Property
~~~~~~~~~~~~~~

.. code-block:: python

  from deepchecks.vision.checks.distribution import ImagePropertyDrift
  from skimage.color import rgb2gray
  import numpy as np


  def aspect_ratio(images: List[np.ndarray]) -> List[float]:
    """Return list of floats of image height to width ratio."""
    return [x[0] / x[1] for x in _sizes(batch)]

  def brightness(images: List[np.ndarray]) -> List[float]:
    """Calculate brightness on each image in the batch."""
    # If grayscale
    if images[0].shape[2] == 1:
        return [img.mean() for img in batch]
    else:
        return [rgb2gray(img).mean() for img in batch]


    properties = [
    {'name': 'Aspect Ratio', 'method': aspect_ratio, 'output_type': 'continuous'},
    {'name': 'Brightness', 'method': brightness, 'output_type': 'continuous'}
  ]

  check = ImagePropertyDrift(alternative_image_properties=properties)


Label Property
~~~~~~~~~~~~~~
For label property the input is also differed by the task type you are running. In this example we implement
properties which are valid for Detection task type.

.. code-block:: python

  from deepchecks.vision.checks.distribution import TrainTestLabelDrift
  from itertools import chain
  import torch


  def number_of_labels(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the number of detections per sample in batch."""
    return [label.shape[0] for label in labels]

  def classes_in_labels(labels: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    classes = [label.reshape((-1, 5))[:, 0].tolist() for label in labels]
    return list(chain.from_iterable(classes))


    properties = [
    {'name': 'Labels Per Sample', 'method': number_of_labels, 'output_type': 'discrete'},
    {'name': 'Classes Appearance', 'method': classes_in_labels, 'output_type': 'class_id'}
  ]

  check = TrainTestLabelDrift(alternative_label_properties=properties)


Prediction Property
~~~~~~~~~~~~~~~~~~~
Prediction property's input, like label property, is also differed by the task type you are running. In this example we
implement properties which are valid for Detection task type.

.. code-block:: python

  from deepchecks.vision.checks.distribution import TrainTestPredictionDrift
  from itertools import chain
  import torch

  def classes_of_predictions(predictions: List[torch.Tensor]) -> List[int]:
    """Return a list containing the classes in batch."""
    classes = [tensor.reshape((-1, 6))[:, -1].tolist() for tensor in predictions]
    return list(chain.from_iterable(classes))

  def bbox_area(predictions: List[torch.Tensor]) -> List[int]:
    """Return a list containing the area of bboxes per image in batch."""
    areas = [(prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).tolist()
             for prediction in predictions]
    return list(chain.from_iterable(areas))


  properties = [
    {'name': 'Classes in Predictions', 'method': classes_of_predictions, 'output_type': 'class_id'},
    {'name': 'Bounding Box Area', 'method': bbox_area, 'output_type': 'continuous'}
  ]

  check = TrainTestPredictionDrift(alternative_prediction_properties=properties)
