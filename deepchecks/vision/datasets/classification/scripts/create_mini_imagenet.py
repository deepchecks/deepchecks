import os, os.path as osp
import shutil
from collections import defaultdict
import numpy as np
import tqdm
from torchvision.datasets import ImageFolder

EXAMPLE_PER_CLASS_VAL = 20
EXAMPLE_PER_CLASS_TRAIN = 30
IMAGENET_ROOT_DIR = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Data/CLS-LOC"
IMAGENET_TARGET_DIR = "/Users/nirbenzvi/code/DeepChecks/ImageNet/Subset"

target_val = osp.join(IMAGENET_TARGET_DIR, "val")
target_train = osp.join(IMAGENET_TARGET_DIR, "train")
val_dataset = ImageFolder(root=osp.join(IMAGENET_ROOT_DIR, "val"))
train_dataset = ImageFolder(root=osp.join(IMAGENET_ROOT_DIR, "train"))

targets = {"train": {"dataset": train_dataset, "counter": defaultdict(list), "list": [], "target_dir": target_train,
                     "max_cls": EXAMPLE_PER_CLASS_TRAIN},
           "val": {"dataset": val_dataset, "counter": defaultdict(list), "list": [], "target_dir": target_val,
                   "max_cls": EXAMPLE_PER_CLASS_VAL}}

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
        [shutil.copy2(img, class_target_dir) for img in img_list]
        target_dict["list"].extend(img_list)

    # write file list
    with open(osp.join(IMAGENET_TARGET_DIR, f"{target_name}_file_list.txt"), 'w') as fid:
        fid.write("\n".join(target_dict["list"]))

