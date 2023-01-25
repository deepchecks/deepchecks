.. _vision_data_class:

=====================
The Vision Data Class
=====================
The :class:`VisionData <deepchecks.vision.VisionData>` data class is the deepchecks base class for
storing your data for a vision task. It is essentially a wrapper around a batch loader of images, labels,
and predictions, that allows deepchecks to efficiently calculate different
:doc:`checks </checks_gallery/vision>` on your data, by caching some of the information.

Information about the supported task types and the required formats for each task is available at
:doc:`/user-guide/vision/supported_tasks_and_formats`.


This file contain three main sections:

* `Common Class Parameters <#common-class-parameters>`__
* `Creating a VisionData Object <#creating-a-visiondata-object>`__
* `Adding Model Predictions <#adding-model-predictions>`__


Common Class Parameters
=======================

- **batch_loader** - The batch loader can be either a pytorch DataLoader, a tensorflow Dataset or any custom
  generator. It is important to note that the data loaded by the batch loader must be **shuffled**.

- **task_type** - The task type of the data, can be either ``classification``, ``object_detection``,
  ``semantic_segmentation``, or ``other``. Data format validation is done upon creation of VisionData based
  on the selected task type. See the :doc:`supported tasks and formats </user-guide/vision/supported_tasks_and_formats>`
  section for more information.

- **label_map** - A dictionary mapping class ids to their names.

- **reshuffle_data** - Whether to reshuffle the data. Since data must be shuffled for the checks to
  work properly, only set this to False if you are sure that the data is already shuffled.

To see all other class parameters, see the :class:`VisionData <deepchecks.vision.VisionData>` API reference.

Creating a VisionData Object
============================
In the sub-sections below we will go over three different ways to create a VisionData object:
1. `From a Generic Generator <#from-a-generic-generator>`__
2. `From a PyTorch DataLoader <#from-pytorch-dataloader>`__
3. `From a TensorFlow Dataset <#from-tensorflow-dataset>`__

The sub-sections contain simple examples for how to create a VisionData object without predictions, in
order to learn how to supply them see the section about :ref:`adding model predictions <vision_data__adding_predictions>`.

From a Generic Generator
----------------------------
If you are not already using a pytorch DataLoader or a tensorflow Dataset for the project, for example
if you are using fastai, jax or any autoML framework, this is the most recommended option.
The custom generator can be implemented in any way you like, as long as it outputs the data in the
:doc:`required format </user-guide/vision/supported_tasks_and_formats>`
and that it loads the data in a shuffled manner. The following is an example of a custom generator
based on data that is fully stored in memory as numpy arrays.

For an example of a custom generator that loads the data from disk batch by batch see the
`following section <#pre-calculated-predictions>`__. A full code implementation of this
method for the MNIST dataset can be seen at
:mod:`deepchecks.vision.datasets.classification.mnist_tensorflow`.


.. code-block:: python

    from deepchecks.vision import VisionData, BatchOutputFormat

    def custom_generator(batch_size = 64):
        images, labels = load_data(shuffle=True)
        for i in range(0, len(images), batch_size):
            # Extracting images and label for batch and converting images of (N, C, H, W) into (N, H, W, C)
            images_in_batch = images[i:i+batch_size].transpose((0, 2, 3, 1))
            labels_in_batch = labels[i:i+batch_size]
            # Convert ImageNet format images into to [0, 255] range format images.
            mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            images_in_batch = np.clip(std * images_in_batch + mean, 0, 1) * 255
            yield BatchOutputFormat(images=images_in_batch, labels=labels_in_batch)

    # Since the data is loaded is a shuffled manner, we do not need to reshuffle it.
    vision_data = VisionData(custom_generator(), task_type='classification', reshuffle_data=False)
    # Visualize the data and verify it is in the correct format
    vision_data.head()

From Pytorch DataLoader
-----------------------
In order to create a VisionData object from a
`PyTorch DataLoader <https://pytorch.org/tutorials/beginner/basics/data_tutorial.html>`_,
all you need is to replace the default
`collate function <https://pytorch.org/docs/stable/data.html#working-with-collate-fn>`_.

The collate function receives a list containing the results of running your implemented
`Dataset's <https://pytorch.org/docs/stable/data.html?highlight=dataset#torch.utils.data.Dataset>`_
``__getitem__`` function on several indexes and returns a batch in any desired format.

In order create a deepchecks compatible DataLoader, you need to create a collate function that
returns a batch in the :doc:`following format </user-guide/vision/supported_tasks_and_formats>`
and replace the default collate function via the ``collate_fn`` argument in the creation of the DataLoader.

A full code implementation of this method for the COCO128 dataset can be seen at
:mod:`deepchecks.vision.datasets.detection.coco_torch`.

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from deepchecks.vision import VisionData, BatchOutputFormat

    def deepchecks_collate(data) -> BatchOutputFormat:
        # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
        images = torch.stack([x[0] for x in data]).permute(0, 2, 3, 1)
        labels = [x[1] for x in data]
        # Convert ImageNet format images into to [0, 255] range format images.
        mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        images = np.clip(std * images.numpy() + mean, 0, 1) * 255
        return BatchOutputFormat(images= images, labels= labels)

    data_loader = DataLoader(my_dataset, batch_size=64, collate_fn=deepchecks_collate)
    vision_data = VisionData(data_loader, task_type='classification')
    # Visualize the data and verify it is in the correct format
    vision_data.head()

From TensorFlow Dataset
-----------------------
There are two possible ways to create a deepchecks compatible tensorflow
`Dataset object <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_. You can either create it
in a way that directly outputs the data in the required format or convert an existing dataset.
We will demonstrate the second option.

In the following example, we have a tensorflow dataset object that outputs a
batch of images and labels as a tuple of (images, labels).
We will use the `map <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map>`_
function to convert the data into :doc:`Deepchecks' format </user-guide/vision/supported_tasks_and_formats>`.

A full code implementation of this method for the COCO128 dataset can be seen at the following
`link <https://github.com/deepchecks/deepchecks/tree/main/deepchecks/vision/datasets/detection/coco_tensorflow.py>`_.

.. code-block:: python

    from deepchecks.vision import VisionData, BatchOutputFormat

    def deepchecks_map(batch) -> BatchOutputFormat:
        # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
        images = batch[0].permute(0, 2, 3, 1)
        labels = batch[1]
        # Convert ImageNet format images into to [0, 255] range format images.
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        images = np.clip(std * images.numpy() + mean, 0, 1) * 255
        return BatchOutputFormat(images= images, labels= labels)

    deepchecks_dataset = my_dataset.map(deepchecks_map)
    vision_data = VisionData(deepchecks_dataset, task_type='classification')
    # Visualize the data and verify it is in the correct format
    vision_data.head()

.. _vision_data__adding_predictions:
Adding Model Predictions
========================
Some checks, including the :doc:`model evaluation checks and suite </checks_gallery/vision>`,
require model predictions in
order to run. Model predictions are supplied via the batch loader in a similar fashion to the images and labels.
There are several ways to supply them which can be roughly divide into two categories: Pre-calculated predictions and
on-demand inference.

.. _vision_data__precalculated_predictions:
Pre-calculated Predictions
--------------------------
It is recommended to use this option if your model object is unavailable locally (for example if
placed on a separate prediction server) or if the predicting process is computationally expensive or time consuming.

In the example below we will read the pre-calculated predictions, as well as the images and labels, from
a csv file containing the path to the image, the label and the prediction probabilities per sample.

.. code-block:: python

    from PIL import Image
    from deepchecks.vision import VisionData, BatchOutputFormat

    def data_from_file_generator(batch_size = 64):
        data = pd.read_csv('classification_data.csv', index_col=0)
        # Shuffling is a must for generic generators in order to achieve accurate results
        data = data.sample(frac=1)
        for i in range(0, len(data), batch_size):
            images = [Image.open(x) for x in data['path_to_image'][i:(i + batch_size):]]
            labels = data['label'][i:(i + batch_size):]
            prediction_probabilities_as_str = data['prediction_probabilities'][i:(i + batch_size):]
            prediction_probabilities_as_arr = [x.strip('][').split(', ') for x in prediction_probabilities_as_str]
            yield BatchOutputFormat(images= images, labels=labels,
                                    predictions= np.array(prediction_probabilities_as_arr, dtype=np.float32))

    # Since the data is shuffled beforehand, we do not need to reshuffle it.
    vision_data = VisionData(data_from_file_generator(), task_type='classification', reshuffle_data=False)
    # Visualize the data and verify it is in the correct format
    vision_data.head()

On-demand Inference
-------------------
In this case we will need to incorporate the model object in the relevant format transformation function
(the ``collate`` function for pytorch or the ``map`` function for tensorflow). This can be
done either by using the model as a global variable, creating a wrapper class for the transformation function or
creating a closure function.
We will demonstrate the last option via the pytorch interface.

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from deepchecks.vision import VisionData, BatchOutputFormat

    def create_deepchecks_collate(model, device):
        def deepchecks_collate(data) -> BatchOutputFormat:
            # Extracting images and label and predicting using the model
            raw_images = torch.stack([x[0] for x in data])
            predictions = model(images.to(device)).detach()
            labels = [x[1] for x in data]
            # Convert ImageNet format images into to [0, 255] range format images.
            mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            images = np.clip(std * raw_images.permute(0, 2, 3, 1).numpy() + mean, 0, 1) * 255
            return BatchOutputFormat(images= images, labels= labels, predictions= predictions)
        return deepchecks_collate

    data_loader = DataLoader(my_dataset, batch_size=64,
                             collate_fn=create_deepchecks_collate(my_model, device))
    vision_data = VisionData(data_loader, task_type='classification')
    # Visualize the data and verify it is in the correct format
    vision_data.head()


