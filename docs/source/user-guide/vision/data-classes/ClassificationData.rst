.. _classification_data_class:

=============================
The Classification Data Class
=============================

The ClassificationData class represents a CV classification task in deepchecks.
It is a subclass of the :class:`~deepchecks.vision.VisionData` class and is used to load and preprocess data for a
classification task.
The ClassificationData class is containing additional data and general methods intended for easily accessing metadata
relevant for validating a computer vision classification ML models.

For more info, please visit the API reference page: :class:`~deepchecks.vision.ClassificationData`

Accepted Image Format
---------------------
All checks in deepchecks require images in the same format. They use the `batch_to_images` function in order to get
the images in the correct format. For more info on the accepted formats, please visit the
`VisionData User Guide <VisionData.rst>`_.

Accepted Label Format
---------------------
Deepchecks' checks use the `batch_to_labels` function in order to get the labels in the correct format.
The accepted label format for the ClassificationData is a tensor of shape (N,), when N is the number of samples.
Each element is an integer representing the class index. For example, for a task with 3 different classes, the label
tensor could be: ``[0, 1, 2]``.

Accepted Prediction Format
--------------------------
Deepchecks' checks use the `infer_on_batch` function in order to get the predictions of the model in the correct format.
The accepted prediction format for classification is a tensor of shape (N, n_classes), where N is the number of
samples. Each element is an array of length n_classes that represent the probability of each class. For example, for a
task with 3 different classes, the prediction tensor could be: ``[[0.1, 0.2, 0.7], [0.2, 0.3, 0.5], [0.3, 0.4, 0.1]]``.

Examples
--------

.. code-block:: python

    from deepchecks.vision import ClassificationData
    import torch.nn.functional as F

    class MyClassificationTaskData(ClassificationData)
    """Implement a ClassificationData class for a classification task."""

        def batch_to_images(self, batch):
            """Convert a batch of images to a list of PIL images.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of images to convert.

            Returns
            -------
            list
                A list of PIL images.
            """

            # Assuming batch[0] is a batch of (N, C, H, W) images, we convert it to (N, H, W, C)/
            imgs = batch[0].detach().numpy().transpose((0, 2, 3, 1))

            # The images are normalized to [0, 1] range based on the mean and std of the ImageNet dataset, so we need to
            # convert them back to [0, 255] range.
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
            imgs = std * imgs + mean
            imgs = np.clip(imgs, 0, 1)
            imgs *= 255

            return imgs

        def batch_to_labels(self, batch):
            """Convert a batch of labels to a tensor.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of labels to convert.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N,).
            """

            return batch[1]

        def infer_on_batch(self, batch, model, device):
            """Get the predictions of the model on a batch of images.

            Parameters
            ----------
            batch : torch.Tensor
                The batch of data.
            model : torch.nn.Module
                The model to use for inference.
            device : torch.device
                The device to use for inference.

            Returns
            -------
            torch.Tensor
                A tensor of shape (N, n_classes).
            """

            # Assuming the model returns the logits of the predictions, we need to convert them to probabilities.
            logits = model.to(device)(batch[0].to(device))
            return F.softmax(logits, dim=1)

    # Now, in order to test the class, we can create an instance of it:
    data = MyClassificationTaskData(your_dataloader)

    # And validate the implementation:
    data.validate()

