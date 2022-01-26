import albumentations as A
from deepchecks.vision.base import VisionDataset
from deepchecks.vision.checks.performance import PerformanceReport
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


# Now we duplicate the val_dataloader and create an augmented one
# Note that p=1.0 since we want to apply those to entire dataset
# To use albumentations I need to do this:
augmentaions = [
    A.RandomBrightnessContrast(p=1.0),
    A.ShiftScaleRotate(p=1.0),
    A.HueSaturationValue(p=1.0),
    A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
]
# We put a NoOp for first spot (e.g. identity) so we can replace first Op at every iteration
val_dataloader.dataset.transform = A.Compose([A.NoOp()] + val_dataloader.dataset.transform.transforms.transforms)
augmented_dataloader.dataset.trasform = A.Compose([A.NoOp()] + augmented_dataloader.dataset.transform.transforms.transforms)
for a in augmentaions:
    # We will override the first augmentation, the one that is currently identity, with the one we want to test for
    # Robustness
    curr_img_transform = augmented_dataloader.dataset.trasform
    augmented_dataloader.dataset.transform = A.Compose([a] + curr_img_transform.transforms[1:])
    # Create two Check loaders
    baseline_ds = VisionDataset(val_dataloader)
    aug_ds = VisionDataset(augmented_dataloader)

    # Finally run the Check
    check = PerformanceReport() # label_map=class_names
    result = check.run(baseline_ds, aug_ds, model)
    pass