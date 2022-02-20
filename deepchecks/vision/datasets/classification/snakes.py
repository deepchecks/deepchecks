import os.path

# Local
from torchvision.datasets.utils import download_and_extract_archive

from deepchecks.vision.utils.image_utils import AlbumentationImageFolder

def load_data():
    data_dir = os.path.expanduser("~/code/DeepChecks/Datasets/snakes_train_val/val")
    return AlbumentationImageFolder(root=data_dir)

def _download_snake_data(snake_data_url: str, snake_train_list: str, snake_val_list: str, data_dir: str):
    # TODO implement this; get dataset from Kaggle and use supplied lists to split it online
    # download_and_extract_archive(snake_data_url, './', data_dir)
    pass