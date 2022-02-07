from bs4 import BeautifulSoup
import torchvision
from torchvision import transforms
import torch
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os

dataset_path = "./face-mask-detection"
batch_size = 64
num_classes = 3

from masks_utils import generate_box, generate_label, generate_target

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

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return self.num_imgs

data_transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)


dataset = MaskDataset(dataset_path, data_transform)

train_num = int(len(dataset) / 10 * 9)
train_dataset, test_dataset = torch.utils.data.random_split(
    dataset, [train_num, len(dataset) - train_num], generator=torch.Generator().manual_seed(42)
)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, collate_fn=(lambda batch: tuple(zip(*batch))))
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, collate_fn=(lambda batch: tuple(zip(*batch))))

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
in_features = model.roi_heads.box_predictor.cls_score.in_features
# replace the pre-trained head with a new one
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

load_model_path = './model_epoch_17_loss_64.08599090576172.ckpt'
checkpoint = torch.load(load_model_path, map_location=device)

model.load_state_dict(checkpoint["model_state_dict"])
optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
start_epoch_ind = checkpoint["epoch_ind"]

model.eval()


def label_transformer(annotation_dict_tuple):
    return_list = []
    if isinstance(annotation_dict_tuple, dict):
        annotation_dict_tuple = [annotation_dict_tuple]
    for annotation_dict in annotation_dict_tuple:
        x1y1x2y2_tensor = annotation_dict["boxes"]
        xy_tensor = x1y1x2y2_tensor[:, :2]
        wh_tensor = x1y1x2y2_tensor[:, 2:4] - x1y1x2y2_tensor[:, :2]

        label_tensor = annotation_dict["labels"].reshape(-1, 1)

        xywh_label_tensor = torch.hstack([label_tensor, xy_tensor, wh_tensor])

        return_list.append(xywh_label_tensor)

    return return_list


def prediction_extract(annotation_dict_list):
    list_corrected_predictions = []
    for annotation_dict in annotation_dict_list:
        x1y1x2y2_tensor = annotation_dict["boxes"]
        xy_tensor = x1y1x2y2_tensor[:, :2]
        wh_tensor = x1y1x2y2_tensor[:, 2:4] - x1y1x2y2_tensor[:, :2]

        label_tensor = annotation_dict["labels"].reshape(-1, 1)

        confidence_tensor = annotation_dict["scores"].reshape(-1, 1)

        xywh_label_conf_tensor = torch.hstack([xy_tensor, wh_tensor, confidence_tensor, label_tensor])

        list_corrected_predictions.append(xywh_label_conf_tensor)

    return list_corrected_predictions

from deepchecks.vision import VisionDataset
from deepchecks.vision.utils.detection_formatters import DetectionLabelFormatter, DetectionPredictionFormatter
label_formatter = DetectionLabelFormatter(label_transformer)
prediction_formatter = DetectionPredictionFormatter(prediction_extract)

ds_train = VisionDataset(train_loader, num_classes=3, label_transformer=label_formatter)
ds_test = VisionDataset(test_loader, num_classes=3, label_transformer=label_formatter)

from deepchecks.vision.checks.distribution import TrainTestLabelDrift

TrainTestLabelDrift().run(ds_train, ds_test)