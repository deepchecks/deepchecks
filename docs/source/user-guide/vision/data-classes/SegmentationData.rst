.. _detection_data_class:

====================================
The Semantic Segmentation Data Class
====================================

The Segmentation is a :doc:`data class </user-guide/vision/data-classes/index>` designed for semantic segmentation tasks.
It is a subclass of the :class:`~deepchecks.vision.VisionData` class and is used to help deepchecks load and interact with semantic segmentation data using a well defined format.
detection related checks.

For more info, please visit the API reference page: :class:`~deepchecks.vision.SegmentationData`

Accepted Image Format
---------------------

All checks in deepchecks require images in the same format. They use the :func:`~deepchecks.vision.VisionData.batch_to_images` function in order to get
the images in the correct format. For more info on the accepted formats, please visit the
:doc:`VisionData User Guide </user-guide/vision/data-classes/VisionData>`.

Accepted Label Format
---------------------

Deepchecks' checks use the :func:`~deepchecks.vision.SegmentationData.batch_to_labels` function in order to get the labels in the correct format.
The accepted label format is a list of length N containing
tensors of shape (H, W), where N is the number of images, and H and W are the height and width of the
corresponding image, and its values are the true class_ids of the corresponding pixels in that image.
Note that the tensor should 2D, as the number of channels on the original image are irrelevant to the class.

Accepted Prediction Format
--------------------------

Deepchecks' checks use the :func:`~deepchecks.vision.SegmentationData.infer_on_batch` function in order to get the predictions of the model in the correct format.
The accepted prediction format is a list of length N containing
tensors of shape (C, H, W), where N is the number of images, H and W are the height and width of the
corresponding image, and C is the number of classes that can be detected, each channel corresponds to a
class_id.
Note that the values of dimension C are the probabilities for each class and should sum to 1.

Example
--------

Assuming we have implemented a torch DataLoader whose underlying __getitem__ method returns a tuple of the form:
``(images, labels)``. ``images`` is a tensor of shape (N, C, H, W) in which the images pixel values are normalized to
[0, 1] range based on the mean and std of the ImageNet dataset. ``labels`` is a tensor of shape (N, H, W) in which
each pixel is an integer correlating with the relevant class_id.

.. code-block:: python

    from deepchecks.vision import SegmentationData
    import torch
    import numpy as np

    class MySegmentationTaskData(DetectionData)
    """A deepchecks data digestion class for object detection related checks."""

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
            """Convert a batch of labels to the required format.

            Parameters
            ----------
            batch : tuple
                The batch of data, containing images and labels.

            Returns
            -------
            List
                A list of size N containing tensors of shape (H, W).
            """
            # In this example, each image's label is a tensor of boolean masks, one per class_id, indicating whether
            # that pixel is of that class.
            # We would like to convert to a format where the function returns a single mask indicating the exact
            # of each pixel:
            images = batch[0]
            labels = batch[1]
            return_labels = []

            for label, image in zip(images, labels):
                # Here, class_id "0" is "background" or "no class detected"
                ret_label = np.zeros((image.shape[0], image.shape[1]))
                # Mask to mark which pixels are already identified as classes, in case of overlap in boolean masks
                ret_label_taken_positions = np.zeros(ret_label.shape)

                # Go over all masks of this image and transform them to a single one:
                for i in range(len(label)):
                    mask = np.logical_and(np.logical_not(ret_label_taken_positions), np.array(label[i]))
                    ret_label += i * mask

                    # Update the taken positions:
                    ret_label_taken_positions = np.logical_or(ret_label_taken_positions, mask)
                return_labels.append(ret_label)

            return return_labels


        def infer_on_batch(self, batch, model, device):
            """Get the predictions of the model on a batch of images.

            Parameters
            ----------
            batch : tuple
                The batch of data, containing images and labels.
            model : torch.nn.Module
                The model to use for inference.
            device : torch.device
                The device to use for inference.

            Returns
            -------
            List
                A list of size N containing tensors of shape (C, H, W).
            """

            # Converts prediction received as (H, W, C) format to (C, H, W) format:
            return_list = []

            predictions = model(batch[0])
            for single_image_tensor in predictions:
                single_image_tensor = torch.transpose(single_image_tensor, 0, 2)
                single_image_tensor = torch.transpose(single_image_tensor, 1, 2)
                return_list.append(single_image_tensor)

            return return_list

    # Now, in order to test the class, we can create an instance of it:
    data = MySegmentationTaskData(your_dataloader)

    # And validate the implementation:
    data.validate_format(your_model)

