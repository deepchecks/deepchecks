import json
import os, os.path as osp
from typing import Sequence, Optional, Union
import datetime
import cv2
import numpy as np
import uuid

YOLO_PATH = "/Users/nirbenzvi/code/DeepChecks/coco128"
# Complete this by putting COCO labels
CATEGORIES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

class YoloParser:
    def __init__(self, category_list: Optional[Union[Sequence, str, dict]] = CATEGORIES):
        if isinstance(category_list, (list, dict, tuple)):
            self._categories = category_list
        else:
            with open(category_list, 'r') as fid:
                self._categories = fid.readlines()
        self._annotations = []
        self._images = []

    def parse_label_file(self, full_label_path: str):
        labels = []
        with open(full_label_path, 'r') as fid:
            for line in fid:
                labels.append(list(map(float, line.split(" "))))
        return np.array(labels)

    def parse_images_and_labels(self, images_path: str, labels_path: str):
        """
        We assume image and labels are correlated, meaning equivalent directories with matching
        image and label names
        Args:
            images_path:
            labels_path:

        Returns:

        """
        for img_path in [f for f in os.listdir(images_path) if f[-3:].lower() in ["jpg", "jpeg", "png"]]:
            full_img_path = osp.join(images_path, img_path)
            full_label_path = osp.join(labels_path, osp.splitext(img_path)[0] + ".txt")
            assert osp.isfile(full_label_path), f"No matching label for image {full_img_path}!"
            h, w, _ = cv2.imread(full_img_path).shape
            labels = self.parse_label_file(full_label_path)
            if len(labels):
                labels[:, [1, 3]] *= w
                labels[:, [2, 4]] *= h
            # TODO perhaps running index?
            curr_img_id = uuid.uuid4().int
            img_dict = {"id": curr_img_id,
                        "license": -1,
                        "coco_url": "N/A",
                        "flickr_url": "N/A",
                        "width": w,
                        "height": h,
                        "file_name": full_img_path,
                        "date_captured": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")},
            self._images.append(img_dict)
            for l in labels:
                bbox = l[1:].tolist()
                category_id = int(l[0])
                # annotation ID doesn't really matter so we use running index
                img_ann = {"id": len(self._annotations),
                           "category_id": category_id,
                           "iscrowd": 0,
                           "segmentation": [[]],
                           "image_id": curr_img_id,
                           "area": bbox[2] * bbox[3],
                           "bbox": bbox}
                self._annotations.append(img_ann)


    def parse_yolo_dir(self, yolo_path: str):
        """
        Creates COCO dataset;
        used https://docs.aws.amazon.com/rekognition/latest/customlabels-dg/md-coco-overview.html
        for reference
        Args:
            yolo_path:

        Returns:

        """
        image_dir = osp.join(yolo_path, "images")
        label_dir = osp.join(yolo_path, "labels")
        if not osp.isdir(image_dir) or not osp.isdir(label_dir):
            raise RuntimeError("Bad YOLO directory structure")
        dataset_subdirs = [f for f in os.listdir(image_dir) if osp.isdir(osp.join(image_dir, f))]
        inter_label_dict = {}
        for subdir in dataset_subdirs:
            # this implies image directory with corresponding labels
            if osp.isdir(osp.join(label_dir, subdir)):
                c_image_dir = osp.join(image_dir, subdir)
                c_label_dir = osp.join(label_dir, subdir)
                self.parse_images_and_labels(c_image_dir, c_label_dir)

    def save_coco_json(self, output_path: str):
        coco_json = {}
        coco_json["info"] = {
            "description": "COCO Dataset From Script",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": datetime.datetime.now().date().year,
            "contributor": "@nirbenz",
            "date_created": datetime.datetime.now().date().strftime("%Y/%m/%d")
        }
        # TODO license
        coco_json["licenses"] = "#TODO"
        coco_json["images"] = self._images
        coco_json["annotations"] = self._annotations
        if isinstance(self._categories, dict):
            coco_json["categories"] = self._categories
        else:
            coco_json["categories"] = [{"id": idx, "supercategory": c, "name": c} for idx, c in  enumerate(self._categories)]
        with open(output_path, 'w') as fid:
            json.dump(coco_json, fid, indent=4, sort_keys=True)

if __name__ == '__main__':
    parser = YoloParser()
    parser.parse_yolo_dir(YOLO_PATH)
    parser.save_coco_json("./coco_128.json")