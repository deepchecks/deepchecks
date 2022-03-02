from abc import abstractmethod
from typing import Optional, Dict

from torch.utils.data import DataLoader
from deepchecks.vision.dataset import TaskType

from deepchecks.vision.vision_task import VisionTask


class ClassificationTask(VisionTask):

    def __init__(self,
                 data_loader: DataLoader,
                 num_classes: Optional[int] = None,
                 label_map: Optional[Dict[int, str]] = None,
                 sample_size: int = 1000,
                 random_seed: int = 0,
                 transform_field: Optional[str] = 'transforms'):

        super().__init__(data_loader, num_classes, label_map, sample_size,
                         random_seed, transform_field)
        self.task_type = TaskType.CLASSIFICATION

    @abstractmethod
    def batch_to_images(self, batch):
        raise NotImplementedError(
            "batch_to_images() must be implemented in a subclass"
        )


