import os
import re
import shutil
from collections import Counter

import tqdm

imagenet_home = "path to imagenet"
imagenet_val_dir = os.path.join(imagenet_home, "Data/CLS-LOC/val/done")
imagenet_val_map = os.path.join(imagenet_home, "ILSVRC2012_validation_ground_truth.txt")
imagenet_synsets = os.path.join(imagenet_home, "ILSVRC2012_mapping.txt")

with open(imagenet_synsets, 'r') as fid:
    imagenet_synsets_map = {int(l.split()[0]): l.split()[1] for l in fid}

with open(imagenet_val_map, 'r') as fid:
    imagenet_gt_map = {idx + 1: int(l.strip()) for idx, l in enumerate(fid)}

synset_cap = Counter()

os.makedirs(os.path.join(imagenet_val_dir, "done"), exist_ok=True)
for val_img in tqdm.tqdm([f for f in os.listdir(imagenet_val_dir) if f[-4:] == "JPEG"]):
    full_img_path = os.path.join(imagenet_val_dir, val_img)
    idx = int(re.match(r"ILSVRC2012_val_([\d]+).JPEG", val_img).group(1))
    label = imagenet_gt_map[idx]
    synset = imagenet_synsets_map[label]
    synset_cap[synset] += 1
    os.makedirs(os.path.join(imagenet_val_dir, synset), exist_ok=True)
    shutil.copy2(full_img_path, os.path.join(imagenet_val_dir, synset))
    shutil.move(full_img_path, os.path.join(imagenet_val_dir, "done"))