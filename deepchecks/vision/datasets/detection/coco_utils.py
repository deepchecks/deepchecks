# ----------------------------------------------------------------------------
# Copyright (C) 2021-2022 Deepchecks (https://www.deepchecks.com)
#
# This file is part of Deepchecks.
# Deepchecks is distributed under the terms of the GNU Affero General
# Public License (version 3 or later).
# You should have received a copy of the GNU Affero General Public License
# along with Deepchecks.  If not, see <http://www.gnu.org/licenses/>.
# ----------------------------------------------------------------------------
#
"""Util Module."""
import os
import pathlib
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import cv2
import numpy as np
from PIL import Image

COCO_DIR = pathlib.Path(__file__).absolute().parent.parent / 'assets' / 'coco_detection'


def download_coco128(root: Path):
    """Download coco from ultralytics using torchvision download_and_extract_archive."""
    root = root if isinstance(root, Path) else Path(root)
    coco_dir = root / 'coco128'
    images_dir = coco_dir / 'images' / 'train2017'
    labels_dir = coco_dir / 'labels' / 'train2017'

    if not (root.exists() and root.is_dir()):
        raise RuntimeError(f'root path does not exist or is not a dir - {root}')

    if images_dir.exists() and labels_dir.exists():
        return coco_dir, 'train2017'
    with urlopen('https://ndownloader.figshare.com/files/37681632') as http_response:
        with ZipFile(BytesIO(http_response.read())) as zip_file:
            zip_file.extractall(path=root)

    # Removing the README.txt file if it exists since it causes issues with sphinx-gallery
    try:
        os.remove(str(coco_dir / 'README.txt'))
    except FileNotFoundError:
        pass

    return coco_dir, 'train2017'


def get_image_and_label(image_file, label_file, transforms=None):
    """Get image and label in correct format for models from file paths."""
    opencv_image = cv2.imread(str(image_file))
    pil_image = Image.fromarray(cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB))
    if label_file is not None and label_file.exists():
        img_labels = [l.split() for l in label_file.open('r').read().strip().splitlines()]
        img_labels = np.array(img_labels, dtype=np.float32)
    else:
        img_labels = np.zeros((0, 5), dtype=np.float32)

    # Transform x,y,w,h in yolo format (x, y are of the image center, and coordinates are normalized) to standard
    # x,y,w,h format, where x,y are of the top left corner of the bounding box and coordinates are absolute.
    bboxes = []
    for label in img_labels:
        x, y, w, h = label[1:]
        # Note: probably the normalization loses some accuracy in the coordinates as it truncates the number,
        # leading in some cases to `y - h / 2` or `x - w / 2` to be negative
        bboxes.append(np.array([
            max((x - w / 2) * pil_image.width, 0),
            max((y - h / 2) * pil_image.height, 0),
            w * pil_image.width,
            h * pil_image.height,
            label[0]
        ]))

    if transforms is not None:
        # Albumentations accepts images as numpy and bboxes in defined format + class at the end
        transformed = transforms(image=np.array(pil_image), bboxes=bboxes)
        pil_image = Image.fromarray(transformed['image'])
        bboxes = transformed['bboxes']

    return pil_image, bboxes


LABEL_MAP = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}
