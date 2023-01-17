.. _vision_data_class:

========================
The Vision Data Class
========================
The :class:`VisionData <deepchecks.vision>` data class is the base deepchecks class for
representing the data for a vision task. It is essentially a wrapper around a batch loader of images, labels,
and predictions in :doc:`Deepchecks' format </user-guide/vision/supported_tasks_and_formats>`
which is used as the input for the different :doc:`vision checks </checks_gallery/vision>`.
In addition it stores cache regarding the data, labels and predictions it encountered during the check run.


Class Properties
================

The common properties are:

- **batch_loader** - The batch loader can be either a pytorch DataLoader, a tensorflow Dataset or any custom
  generator. It is important to note that the data loaded by the
  batch loader iterator must be **shuffled**.

- **task_type** - The task type of the data, can be either ``classification``, ``object_detection``,
  ``semantic_segmentation``, or ``other``. Data format validation is done upon creation of VisionData based
  on the selected task type.

- **label_map** - A dictionary mapping class ids to their names.

Creating a VisionData Object
============================

From Generic Generator
----------------------
If you are not already using pytorch or tensorflow for the project, this is the most recommended option.
The custom generator can be implemented in any way you like, as long as it outputs the data in the
:doc:`required format </user-guide/vision/supported_tasks_and_formats>`
and that it loads the data in a shuffled manner. The following is an example of a custom generator based on data that
is loaded and stored in memory as numpy arrays.

For an example of a custom generator that loads the data from disk batch by batch see the
:ref:`following section <vision_data__precalculated_predictions>`.


.. code-block:: python

    from deepchecks.vision import VisionData, BatchOutputFormat

    def custom_generator(batch_size = 64):
        images, labels = load_data(shuffle=True)
        for i in range(0, len(images), batch_size):
            # Extracting images and label for batch and converting images of (N, C, H, W) into (N, H, W, C)
            images_in_batch = images[i:i+batch_size].transpose((0, 2, 3, 1))
            labels_in_batch = labels[i:i+batch_size]
            # Convert ImageNet format images into to [0, 255] range format images.
            mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            images_in_batch = np.clip(std * images_in_batch + mean, 0, 1) * 255
            yield BatchOutputFormat(images= images_in_batch, labels= labels_in_batch)

    vision_data = VisionData(custom_generator(), task_type='classification', shuffle_batch_loader=False)
    vision_data.head()

From Pytorch DataLoader
-----------------------
In order to create a pytorch DataLoader, you must first create a pytorch Dataset object (link). When creating a pytorch
Dataset object, you must implement either the ``__get_item__`` function or the ``__iter__`` function, whose output is
a single data element.

Based on the output format we will create a collate function that will group several data
elements into a batch in :doc:`Deepchecks' format </user-guide/vision/supported_tasks_and_formats>`.
This collate function will be used as the ``collate_fn`` argument in the creation of the DataLoader.

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

    data_loader = DataLoader(my_dataset, batch_size=64, generator=torch.Generator(),
                             collate_fn=deepchecks_collate)
    vision_data = VisionData(data_loader, task_type='classification')
    vision_data.head()

From TensorFlow Dataset
-----------------------
There are two possible ways to create a deepchecks compatible tensorflow
`Dataset object <https://www.tensorflow.org/api_docs/python/tf/data/Dataset>`_. You can either create it
in a way that directly outputs the data in the required format or convert an existing dataset,
we will demonstrate the latter.

Assume a tensorflow dataset object that outputs a batch of images and labels as a tuple of (images, labels).
We will use the `map <https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map>`_
function to convert the data into :doc:`Deepchecks' format </user-guide/vision/supported_tasks_and_formats>`.

.. code-block:: python

    from deepchecks.vision import VisionData, BatchOutputFormat

    def deepchecks_map(batch) -> BatchOutputFormat:
        # Extracting images and label and converting images of (N, C, H, W) into (N, H, W, C)
        images = batch[0].permute(0, 2, 3, 1)
        labels = batch[1]
        # Convert ImageNet format images into to [0, 255] range format images.
        mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
        images = np.clip(std * images.numpy() + mean, 0, 1) * 255
        return BatchOutputFormat(images= images, labels= labels)

    deepchecks_dataset = my_dataset.map(deepchecks_map)
    vision_data = VisionData(deepchecks_dataset, task_type='classification')
    vision_data.head()



Adding Model Predictions
========================
Some of deepchecks tests, including the :doc:`model evaluation checks and suite </checks_gallery/vision>`,
require model predictions in
order to run. Models prediction are supplied via the batch loader in a similar fashion to the images and labels.
There are several ways to supply them which can be divide into two categories: Pre-calculated predictions and
on-demand inference.

On-demand Inference
-------------------
In this case we will need to incorporate the model object in the relevant format transformation function. This can be
done either by using the model as a global variable, creating a wrapper class for the transformation function or
creating a closure function. We will demonstrate the last option via the pytorch interface.

.. code-block:: python

    import torch
    from torch.utils.data import DataLoader
    from deepchecks.vision import VisionData, BatchOutputFormat

    def create_deepchecks_collate(model):
        def deepchecks_collate(data) -> BatchOutputFormat:
            # Extracting images and label and predicting using the model
            raw_images = torch.stack([x[0] for x in data])
            predictions = model(images)
            labels = [x[1] for x in data]
            # Convert ImageNet format images into to [0, 255] range format images.
            mean, std  = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
            images = np.clip(std * raw_images.permute(0, 2, 3, 1).numpy() + mean, 0, 1) * 255
            return BatchOutputFormat(images= images, labels= labels, predictions= predictions)
        return deepchecks_collate

    data_loader = DataLoader(my_dataset, batch_size=64, generator=torch.Generator(),
                             collate_fn=create_deepchecks_collate(my_model))
    vision_data = VisionData(data_loader, task_type='classification')
    vision_data.head()

.. _vision_data__precalculated_predictions:
Pre-calculated Predictions
--------------------------
It is specifically recommended to use this option if your model object is unavailable locally (for example if
placed on a separate prediction server) or if the predicting process is computationally expensive or time consuming.

In the example below we will read the pre-calculated predictions, as well as the images and labels, from
a csv file containing the path to the image, the label and the prediction probabilities per sample.

.. code-block:: python

    from PIL import Image
    from deepchecks.vision import VisionData, BatchOutputFormat

    def data_from_file_generator(batch_size = 64):
        # Shuffling is a must for generic generators in order to achieve accurate results
        data = pd.read_csv('classification_data.csv', index_col=0).sample(frac=1)
        for i in range(0, len(data), batch_size):
            images = [Image.open(x) for x in data['path_to_image'][i:(i + batch_size):]]
            labels = data['label'][i:(i + batch_size):]
            prediction_probabilities_as_str = data['prediction_probabilities'][i:(i + batch_size):]
            prediction_probabilities_as_arr = [x.strip('][').split(', ') for x in prediction_probabilities_as_str]
            yield BatchOutputFormat(images= images, labels=labels,
                                    predictions= np.array(prediction_probabilities_as_arr, dtype=np.float32))

    vision_data = VisionData(data_from_file_generator(), task_type='classification', shuffle_batch_loader=False)
    vision_data.head()
