import os, os.path as osp
import shutil
from collections import defaultdict

import cv2
import numpy as np
import tqdm
from torchvision.datasets import ImageFolder
from joblib import Parallel, delayed

EXAMPLE_PER_CLASS_VAL = 3
EXAMPLE_PER_CLASS_TRAIN = 3
TARGET_WH = (256, 256)
IMAGENET_ROOT_DIR = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Data/CLS-LOC"
IMAGENET_TARGET_DIR = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Subset_256_3"

target_val = osp.join(IMAGENET_TARGET_DIR, "val")
target_train = osp.join(IMAGENET_TARGET_DIR, "train")
val_dataset = ImageFolder(root=osp.join(IMAGENET_ROOT_DIR, "val"))
train_dataset = ImageFolder(root=osp.join(IMAGENET_ROOT_DIR, "train"))

np.random.seed(42)

targets = {"train": {"dataset": train_dataset, "counter": defaultdict(list), "list": [], "target_dir": target_train,
                     "max_cls": EXAMPLE_PER_CLASS_TRAIN},
           "val": {"dataset": val_dataset, "counter": defaultdict(list), "list": [], "target_dir": target_val,
                   "max_cls": EXAMPLE_PER_CLASS_VAL}}


def copy_images(img_list, class_target_dir, target_size_wh=None):
    if target_size_wh is None:
        [shutil.copy2(img_path, class_target_dir) for img_path in img_list]
    # resize to save space
    else:
        Parallel(n_jobs=1)(delayed(_copy_images)(img_path, class_target_dir, target_size_wh) for img_path in img_list)
        # [_copy_images(img_path, class_target_dir, target_size_wh) for img_path in img_list]

def _copy_images(img_path, class_target_dir, target_size_wh=None):
    img = cv2.imread(img_path)
    img_ = image_resize_ar(img, min_edge=target_size_wh[0], max_edge=0)
    cv2.imwrite(osp.join(class_target_dir, osp.basename(img_path)), img_)

def image_resize_ar(image, min_edge=256, max_edge=None, inter=cv2.INTER_LINEAR):
    """
    inter (OpenCV flag): flag that is used to specify the interpolation algorithm. Should be one of:
            cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA, cv2.INTER_LANCZOS4.
            Default: cv2.INTER_LINEAR.
    TorchVision uses PIL's InterpolationMode.BILINEAR which is equivalent to cv2's LINEAR
    Albumentation uses cv2's INTER_LINEAR by default
    :param image:
    :param width:
    :param height:
    :param inter:
    :return:
    """
    h, w, _ = image.shape
    smallest_edge = min(h, w)
    # rescale the image so the smallest side is min_side
    scale = min_edge / smallest_edge

    # check if the largest side is now greater than max_side, which can happen
    # when images have a large aspect ratio
    largest_edge = max(h, w)
    if max_edge and largest_edge * scale > max_edge:
        scale = max_edge / largest_edge

    # resize the image with the computed scale
    resized = cv2.resize(image, dsize=None, fx=scale, fy=scale, interpolation=inter)

    # return the resized image
    return resized

for target_name, target_dict in targets.items():
    for idx, (img_path, cls_idx) in tqdm.tqdm(enumerate(target_dict["dataset"].samples)):
        # drop
        if len(target_dict["counter"][cls_idx]) < target_dict["max_cls"]:
            target_dict["counter"][cls_idx].append(img_path)
        else:
            """
            j := randomInteger(1, i)
            if j <= k
                R[j] := S[i]
            """
            j = np.random.randint(0, target_dict["max_cls"])
            target_dict["counter"][cls_idx][j] = img_path

    os.makedirs(target_dict["target_dir"], exist_ok=True)
    for cls_idx, img_list in target_dict["counter"].items():
        class_target_dir = osp.join(target_dict["target_dir"], target_dict["dataset"].classes[cls_idx])
        os.makedirs(class_target_dir, exist_ok=True)
        copy_images(img_list, class_target_dir, target_size_wh=TARGET_WH)
        target_dict["list"].extend(img_list)

    # write file list
    with open(osp.join(IMAGENET_TARGET_DIR, f"{target_name}_file_list.txt"), 'w') as fid:
        fid.write("\n".join(target_dict["list"]))

