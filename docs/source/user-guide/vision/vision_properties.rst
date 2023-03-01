.. _vision_properties_guide:

=================
Vision Properties
=================

Properties are one-dimension values that are extracted from either the images, labels or predictions. For example, an
image property is **brightness**, and a label property is **bounding box area** (for detection tasks).
Deepchecks includes :ref:`built-in properties <Deepchecks' Built-in Properties>` and supports implementing your own
properties.

What Are Properties Used For?
=============================

Properties are used by some of the Deepchecks' checks (e.g. train-test drift), in order to extract meaningful
features from the data, since some computations are difficult to compute directly on the images (for example drift).
Inspecting the distribution of the property's values (e.g. to notice some images are extremely dark,
or that the aspect ratio of images is different between the train and test sets) can help uncover potential problems
in the way that the datasets were built, or hint about the model's expected performance on unseen data.

Example for specific scenarios in which measuring properties may come in handy:

#. **Investigating low test performance** - detecting high drift in certain properties may help you pinpoint the causes of
   the model's lower performance on the test data.
#. **Generalizability on new data** - a drift in significant data properties,
   may indicate lower ability of the model to accurately predict on the new (different) unlabeled data.
#. **Find weak segments** - The properties can be used to segment the data and test for low performing segments.
   If found, the weak segment may indicate a underrepresented segment or an area where the data quality is worse.
#. **Find obscure relations between the data and the targets** - the model training might be affected
   by properties we are not aware of, and that aren't the core attributes of what we are aiming for it to learn.
   For example, in a classification dataset of wolves and dogs photographs, if only wolves are photographed in
   the snow, the brightness of the image may be used to predict the label "wolf" easily. In this case, a model
   might not learn to discern wolf from dog by the animal's characteristics, but by using the background color.


Deepchecks' Built-in Properties
===============================

We divide the properties by the data that they are based on: images, labels or predictions.
You can either use the built-in properties or implement your own ones and pass them to the relevant checks.

Image Properties
------------------

The built-in image properties are:

==============================  ==========
Property name                   What is it
==============================  ==========
Aspect Ratio                    Ratio between height and width of image (height / width)
Area                            Area of image in pixels (height * width)
Brightness                      Average intensity of image pixels. Color channels have different weights according to
                                RGB-to-Grayscale formula
RMS Contrast                    Contrast of image, calculated by standard deviation of pixels
Mean Red Relative Intensity     Mean over all pixels of the red channel, scaled to their relative intensity in
                                comparison to the other channels [r / (r + g + b)].
Mean Green Relative Intensity   Mean over all pixels of the green channel, scaled to their relative intensity in
                                comparison to the other channels [g / (r + g + b)].
Mean Blue Relative Intensity    Mean over all pixels of the blue channel, scaled to their relative intensity in
                                comparison to the other channels [b / (r + g + b)].
==============================  ==========

Label & Prediction Properties
-------------------------------

The built-in label & predictions properties are:

===================================  ==========
Property name                        What is it
===================================  ==========
Samples Per Class                    The classes abundance in the data
Bounding Box Area                    Area of bounding boxes in pixels (height * width) for object detection
Number of Bounding Boxes Per Image   Number of bounding boxes in a single image for object detection
===================================  ==========

Property Structure
====================

All property types have a similar structure, which is a dictionary with 3 keys:

- ``name`` - The name of the property
- ``method`` - The callable function that calculates the property's value. It accepts the relevant data and returns
  the values list.
- ``output_type`` - Relates to the method's return values list, and is one of the following:

  - ``continuous`` - For numeric values with continuous nature
  - ``discrete`` - For numeric values with discrete nature or non-numeric values
  - ``class_id`` - Means the output is of class ids. In this case we will try to translate the ids into their
    corresponding class labels.

Each dictionary is a single property, and the checks accepts a list of those dictionaries. For example:

.. code-block:: python

  def mean_image(images):
    return [image.mean() for image in images]

  properties = [
    {'name': 'My Image Mean', 'method': mean_image, 'output_type': 'continuous'}
  ]



The Method's Input
----------------------

Each property is built for the specific data type that it runs on, and receives its deepchecks-expected format,
as demonstrated in :doc:`/user-guide/vision/supported_tasks_and_formats`.
Note that prediction and label-based properties are not interchangeable due to their slightly different format, even if
they calculate similar values.

The Method's Output
----------------------

Each property function must return a sequence in the same length as the length of the input object. This is used later
in order to couple each sample to its right properties values. In image properties we expect each image to generate a
single property value, which results in a list of primitives types in the same length as the number of images. On the
other hand for label & predictions we allow each one to have multiple primitive values (for example area of bounding
box), which means the returned list may contain either primitives values or a lists of primitive values per
label/prediction.


Customizing the Checks' Properties
==================================
By default, checks using properties will use the :ref:`built-in properties <Deepchecks' Built-in Properties>`.

In order to override the default properties and use custom ones, a list of functions to be calculated on the data during
the check (in the format specified above) should be passed to the check init.
:ref:`Properties format <Property Structure>`.

We will demonstrate the 3 drift checks (for each property type) and implement the properties to pass to it.

Image Property
-----------------

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
----------------

For label property the input varies according to the task type you are running. In this example we implement
properties which apply to the Detection task type.

.. code-block:: python

  from deepchecks.vision.checks.distribution import TrainTestLabelDrift

  def number_of_labels(labels: List[np.ndarray]) -> List[int]:
    """Return a list containing the number of detections per sample in batch."""
    return [label.shape[0] for label in labels]

  def classes_in_labels(labels: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [label.reshape((-1, 5))[:, 0].tolist() for label in labels]


    properties = [
    {'name': 'Labels Per Sample', 'method': number_of_labels, 'output_type': 'discrete'},
    {'name': 'Classes Appearance', 'method': classes_in_labels, 'output_type': 'class_id'}
  ]

  check = TrainTestLabelDrift(label_properties=properties)


Prediction Property
---------------------

Prediction property's input, like label property, also varies by the task type you are running. In this example we
implement properties which apply to the Detection task type.

.. code-block:: python

  from deepchecks.vision.checks.distribution import TrainTestPredictionDrift

  def classes_of_predictions(predictions: List[np.ndarray]) -> List[List[int]]:
    """Return a list containing the classes in batch."""
    return [pred.reshape((-1, 6))[:, -1].tolist() for pred in predictions]

  def bbox_area(predictions: List[np.ndarray]) -> List[List[float]]:
    """Return a list containing the area of bboxes per image in batch."""
    return [(prediction.reshape((-1, 6))[:, 2] * prediction.reshape((-1, 6))[:, 3]).tolist()
             for prediction in predictions]


  properties = [
    {'name': 'Classes in Predictions', 'method': classes_of_predictions, 'output_type': 'class_id'},
    {'name': 'Bounding Box Area', 'method': bbox_area, 'output_type': 'continuous'}
  ]

  check = TrainTestPredictionDrift(prediction_properties=properties)


