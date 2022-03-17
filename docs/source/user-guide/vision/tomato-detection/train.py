import os
from functools import partial
import numpy as np
import torch
from PIL import Image
import xml.etree.ElementTree as ET
import torchvision
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead
import albumentations as A
from albumentations.pytorch import ToTensorV2
import time
import math

class TomatoDataset(Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms

        self.images = list(sorted(os.listdir(os.path.join(root, 'images'))))
        self.annotations = list(sorted(os.listdir(os.path.join(root, 'annotations'))))

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        ann_path = os.path.join(self.root, "annotations", self.annotations[idx])
        img = Image.open(img_path).convert("RGB")
        bboxes = []
        labels = []
        with open(ann_path, 'r') as f:
            tree = ET.parse(f)
            root = tree.getroot()
            size = root.find('size')
            w = int(size.find('width').text)
            h = int(size.find('height').text)

            for obj in root.iter('object'):
                difficult = obj.find('difficult').text
                if int(difficult) == 1:
                    continue
                cls_id = 1
                xmlbox = obj.find('bndbox')
                b = [float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text),
                     float(xmlbox.find('ymax').text)]
                bboxes.append(b)
                labels.append(cls_id)

        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        bboxes = torch.as_tensor(np.array(bboxes), dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)

        if self.transforms is not None:
            res = self.transforms(image=np.array(img), bboxes=bboxes, class_labels=labels)

        target = {
            'boxes': [torch.Tensor(x) for x in res['bboxes']],
            'labels': res['class_labels']
        }

        img = res['image']

        return img, target

    def __len__(self):
        return len(self.images)


data_transforms = A.Compose([
        A.Resize(height=400, width=400),
        A.CenterCrop(height=320, width=320),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['class_labels']))


dataset = TomatoDataset(root='/Users/itaygabbay/Deepchecks/deepchecks/docs/source/user-guide/vision/tomato-detection/data',
                        transforms=data_transforms)
train_set, val_set = torch.utils.data.random_split(dataset,
                                                   [int(len(dataset)*0.9), len(dataset)-int(len(dataset)*0.9)],
                                                   generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))
test_loader = DataLoader(val_set, batch_size=64, collate_fn=(lambda batch: tuple(zip(*batch))))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.ssdlite320_mobilenet_v3_large(pretrained=True)

in_channels = det_utils.retrieve_out_channels(model.backbone, (320, 320))
num_anchors = model.anchor_generator.num_anchors_per_location()
norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

model.head.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, 2, norm_layer)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
# and a learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

if False:
    num_epochs = 15
    loss_list = []
    for epoch in range(num_epochs):
        print('Starting training....{}/{}'.format(epoch + 1, num_epochs))
        loss_sub_list = []
        start = time.time()
        for images, targets in train_loader:
            filtered_images = []
            filtered_targets = []

            for image, t in zip(images, targets):
                if len(t['boxes']) > 0:
                    filtered_images.append(image.to(device))
                    filtered_targets.append({k: torch.stack(v).to(device) for k, v in t.items()})

            images = filtered_images
            targets = filtered_targets

            model.train()
            for target in targets:
                if len(target["boxes"]) != len(target["labels"]):
                    print("Error: Different number of boxes and labels in {}".format(target["image_id"]))
                    exit()

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_sub_list.append(loss_value)

            # update optimizer and learning rate
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            # lr_scheduler.step()
        end = time.time()

        # print the loss of epoch
        epoch_loss = np.mean(loss_sub_list)
        loss_list.append(epoch_loss)
        print('Epoch {}/{} Loss: {:.4f} Time: {:.4f}'.format(epoch + 1, num_epochs, epoch_loss, end - start))

    # save the model
    torch.save(model.state_dict(), 'ssd_model.pth')

    exit(0)

model.load_state_dict(torch.load('ssd_model.pth'))

model.eval()


from deepchecks.vision.detection_data import DetectionData

class TomatoData(DetectionData):

    def batch_to_images(self, batch):
        inp = torch.stack(list(batch[0])).numpy().transpose((0, 2, 3, 1))
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)
        return inp * 255

    def batch_to_labels(self, batch):

        tensor_annotations = batch[1]
        label = []
        for annotation in tensor_annotations:
            if len(annotation["boxes"]):
                bbox = torch.stack(annotation["boxes"])
                bbox[:, 2:] = bbox[:, 2:] - bbox[:, :2]
                label.append(
                    torch.concat([torch.stack(annotation["labels"]).reshape((-1, 1)), bbox], dim=1)
                )
            else:
                label.append(torch.tensor([]))
        return label

    def infer_on_batch(self, batch, model, device):
        nm_thrs = 0.2
        score_thrs = 0.7
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


train_data = TomatoData(train_loader)
test_data = TomatoData(test_loader)

from deepchecks.vision.suites import full_suite

res = full_suite().run(train_data, test_data, model)

res.save_as_html('results.html')