from deepchecks.vision.base import VisionDataset
from deepchecks.vision.checks.performance.robustness_report import RobustnessReport
from deepchecks.vision.datasets.classification.imagenet import get_trained_imagenet_model, \
    get_imagenet_dataloaders_albumentations

synsets = "/Users/nirbenzvi/code/DeepChecks/ImageNet/synsets.txt"
with open(synsets, 'r') as fid:
    real_class_names = {l.split()[0]: " ".join(l.split()[1:]).split(",")[0].strip() for l in fid}

model = get_trained_imagenet_model()
_, val_dataloader = get_imagenet_dataloaders_albumentations()
_, augmented_dataloader = get_imagenet_dataloaders_albumentations()
class_names = val_dataloader.dataset.classes
# this maps real classes on top of imagenet's synset format
class_names = [real_class_names[c] for c in class_names]
baseline_ds = VisionDataset(val_dataloader)
aug_ds = VisionDataset(augmented_dataloader)

# Run Check
check = RobustnessReport()  # label_map=class_names
result = check.run(baseline_dataset=baseline_ds, augmented_dataset=aug_ds, model=model)