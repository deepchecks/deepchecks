import os
from functools import partial
from pathlib import Path
from typing import Literal
import typing as t

import cv2
import numpy as np
import torchvision
from bs4 import BeautifulSoup
import torch
from torchvision.models.detection import ssdlite320_mobilenet_v3_large
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead

from deepchecks.vision import DetectionData
from torch.utils.data.sampler import SubsetRandomSampler, SequentialSampler

import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader

from deepchecks import vision


DATA_DIR = Path(__file__).absolute().parent


def generate_box(obj):
    xmin = int(obj.find("xmin").text)
    ymin = int(obj.find("ymin").text)
    xmax = int(obj.find("xmax").text)
    ymax = int(obj.find("ymax").text)

    return [xmin, ymin, xmax, ymax]


def generate_label(obj):
    if obj.find("name").text == "with_mask":
        return 1
    elif obj.find("name").text == "mask_weared_incorrect":
        return 2
    return 0  # "without_mask"


def generate_target(image_id, file):
    with open(file) as f:
        data = f.read()
    soup = BeautifulSoup(data, "lxml")
    objects = soup.find_all("object")

    boxes = []
    labels = []
    for i in objects:
        boxes.append(generate_box(i))
        labels.append(generate_label(i))
    boxes = torch.as_tensor(boxes, dtype=torch.float32)
    # Labels (In my case, I only one class: target class or background)
    labels = torch.as_tensor(labels, dtype=torch.int64)
    # Tensorise img_id
    img_id = torch.tensor([image_id])
    # Annotation is in dictionary format
    target = {}
    target["boxes"] = boxes
    target["labels"] = labels
    target["image_id"] = img_id

    return target


class MaskDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transforms):
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs_dir_path = os.path.join(dataset_path, "images")
        self.annotation_dir_path = os.path.join(dataset_path, "annotations")
        self.num_imgs = len(os.listdir(self.imgs_dir_path))

    def __getitem__(self, idx):
        # === trick to make dataset iterable
        if idx >= len(self):
            raise IndexError

        # load images ad masks
        file_image = "maksssksksss" + str(idx) + ".png"
        file_label = "maksssksksss" + str(idx) + ".xml"
        img_path = os.path.join(self.imgs_dir_path, file_image)
        label_path = os.path.join(self.annotation_dir_path, file_label)
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        # Generate Label
        target = generate_target(idx, label_path)

        # Protect against bboxes greater than image size
        target["boxes"][:, 2] = torch.minimum(target["boxes"][:, 2], torch.Tensor([img.shape[1] - 1]))
        target["boxes"][:, 3] = torch.minimum(target["boxes"][:, 3], torch.Tensor([img.shape[0] - 1]))

        if len(target["boxes"]) != len(target["labels"]):
            print("Error: Different number of boxes and labels")
            exit()

        img, target["boxes"], target["labels"] = self.apply_transform(
            img, bboxes=target["boxes"], labels=target["labels"]
        )

        if len(target["boxes"]) != len(target["labels"]):
            print("Error: Different number of boxes and labels after augmentation")
            exit()

        return img, target

    def __len__(self):
        return self.num_imgs

    def apply_transform(self, img, bboxes, labels):
        """Implement the transform in a function to be able to override it in tests."""
        if self.transforms is not None:
            # Albumentations accepts images as numpy and bboxes in defined format + class at the end
            transformed = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)
            img = transformed['image']
            bboxes = torch.as_tensor(transformed['bboxes'], dtype=torch.float32).reshape((-1, 4))
            labels = torch.as_tensor(transformed['class_labels'], dtype=torch.int64)

        return img, bboxes, labels


class MaskData(DetectionData):

    def batch_to_images(self, batch):
        return [x.detach().numpy().transpose((1, 2, 0)) * 255 for x in batch[0]]

    def batch_to_labels(self, batch):

        tensor_annotations = batch[1]
        label = []
        for annotation in tensor_annotations:
            bbox = annotation["boxes"]
            bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
            label.append(
                torch.concat([annotation["labels"].reshape((-1, 1)), bbox], dim=1)
            )
        return label

    def infer_on_batch(self, batch, model, device):
        nm_thrs = 0.2
        score_thrs = 0.6
        imgs = list(img.to(device) for img in batch[0])
        with torch.no_grad():
            preds = model(imgs)
        processed_pred = []
        for pred in preds:
            keep_boxes = torchvision.ops.nms(pred['boxes'], pred['scores'], nm_thrs)
            score_filter = pred['scores'][keep_boxes] > score_thrs

            # get the filtered result
            test_boxes = pred['boxes'][keep_boxes][score_filter].reshape((-1, 4))
            test_boxes[:, 2:] = test_boxes[:, 2:] - test_boxes[:, :2]  # xyxy to xywh
            test_labels = pred['labels'][keep_boxes][score_filter]
            test_scores = pred['scores'][keep_boxes][score_filter]

            processed_pred.append(
                torch.concat([test_boxes, test_scores.reshape((-1, 1)), test_labels.reshape((-1, 1))], dim=1))
        return processed_pred


LABEL_MAP = {0: 'No Mask', 1: 'Mask', 2: 'Bad Mask'}


def load_dataset(
        train: bool = True,
        batch_size: int = 32,
        num_workers: int = 0,
        shuffle: bool = True,
        pin_memory: bool = True,
        object_type: Literal['VisionData', 'DataLoader'] = 'DataLoader'
) -> t.Union[DataLoader, vision.VisionData]:

    root = DATA_DIR
    mask_dir = os.path.join(root, '../../../../../../data/face-mask-detection/')

    data_transform = A.Compose(
        [
            A.Normalize(mean=0, std=1),
            ToTensorV2(),
        ],
        A.BboxParams(format='pascal_voc', min_area=12 ** 2, min_visibility=0.9, label_fields=['class_labels'])
    )

    dataset = MaskDataset(mask_dir, data_transform)

    # Split dataset in a way that will generate some drift
    train_idx = []
    test_idx = []
    np.random.seed(42)
    for idx in range(853):
        label = generate_target(idx, f'{mask_dir}/annotations/maksssksksss{idx}.xml')

        # drop too large images all together
        if len(label['labels']) > 70:
            continue
        elif len(label['labels']) > 6:
            # images with more than 6 targets but less than 30 get added to test 1/3 times
            if np.random.randint(3) == 0:
                test_idx.append(idx)
            else:
                # 30 and over targets go all to test
                if len(label['labels']) <= 30:
                    train_idx.append(idx)
                else:
                    test_idx.append(idx)
        else:
            # normal images get added to test 1/5 times
            if np.random.randint(5) == 0:
                test_idx.append(idx)
            else:
                train_idx.append(idx)

    if shuffle:
        train_sampler = SubsetRandomSampler(train_idx)
        test_sampler = SubsetRandomSampler(test_idx)
    else:
        train_sampler = SequentialSampler(train_idx)
        test_sampler = SequentialSampler(test_idx)

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                               collate_fn=(lambda batch: tuple(zip(*batch))),
                                               sampler=train_sampler, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, pin_memory=pin_memory,
                                              collate_fn=(lambda batch: tuple(zip(*batch))),
                                              sampler=test_sampler, num_workers=num_workers)

    if object_type == 'VisionData':
        if train:
            return MaskData(train_loader, label_map=LABEL_MAP, num_classes=len(LABEL_MAP))
        else:
            return MaskData(test_loader, label_map=LABEL_MAP, num_classes=len(LABEL_MAP))
    else:
        if train:
            return train_loader
        else:
            return test_loader


def load_mask_mobilenet_ssd(device: torch.device) -> torch.nn.Module:

    model_file_name = 'aug_model_107_0.5218211120615402.pth'

    model = ssdlite320_mobilenet_v3_large(pretrained=True)
    num_classes = 3  # background, without_mask, with_mask
    in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
    num_anchors = model.anchor_generator.num_anchors_per_location()
    norm_layer = partial(torch.nn.BatchNorm2d, eps=0.001, momentum=0.03)

    model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
    model.load_state_dict(torch.load(os.path.join(DATA_DIR, model_file_name), map_location=device))
    model = model.to(device)
    return model
