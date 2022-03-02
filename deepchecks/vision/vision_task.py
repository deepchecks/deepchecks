from abc import abstractmethod
from typing import Optional, Dict

from torch.utils.data import DataLoader


class VisionTask:
    """VisionTask represent a base task in deepchecks. It wraps PyTorch DataLoader together with model related metadata.

    The VisionTask class is containing additional data and general methods intended for easily accessing
    metadata relevant for validating a computer vision ML models.

    Parameters
    ----------
    data_loader : DataLoader
        PyTorch DataLoader object. If your data loader is using IterableDataset please see note below.
    num_classes : int, optional
        Number of classes in the dataset. If not provided, will be inferred from the dataset.
    label_map : Dict[int, str], optional
        A dictionary mapping class ids to their names.
    sample_size : int, default: 1,000
        Sample size to run the checks on.
    random_seed : int, default: 0
        Random seed used to generate the sample.
    transform_field : str, default: 'transforms'
        Name of transforms field in the dataset which holds transformations of both data and label.

    Notes
    -----
    Accepted label formats are:
        * Classification: tensor of shape (N,), When N is the number of samples. Each element is an integer
          representing the class index.
        * Object Detection: List of length N containing tensors of shape (B, 5), where N is the number of samples,
          B is the number of bounding boxes in the sample and each bounding box is represented by 5 values: (class_id,
          x, y, w, h). x and y are the coordinates (in pixels) of the upper left corner of the bounding box, w and h are
          the width and height of the bounding box (in pixels) and class_id is the class id of the prediction.

    The labels returned by the data loader (e.g. by using next(iter(data_loader))[1]) should be in the specified format,
    or else the callable label_formatter should be able to transform the labels to the desired format.
    """

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 sample_size: int = 1000,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transforms'):

        self.data_loader = data_loader
        self.num_classes = num_classes
        self.label_map = label_map
        self.sample_size = sample_size
        self.random_seed = random_seed
        self.transform_field = transform_field

    @abstractmethod
    def batch_to_images(self, batch):
        raise NotImplementedError(
            "batch_to_images() must be implemented in a subclass"
        )