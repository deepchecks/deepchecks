from bs4 import BeautifulSoup
import torch
import matplotlib.pyplot as plt
from matplotlib import patches


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


def plot_image(img_tensor, tensor_annotation):

    fig, ax = plt.subplots(1)
    img = img_tensor.cpu().data

    # Display the image
    ax.imshow(img.permute(1, 2, 0))

    for box, label in zip(tensor_annotation["boxes"], tensor_annotation["labels"]):
        xmin, ymin, xmax, ymax = box.cpu().data
        if label == 0:
            color = "r"
        elif label == 1:
            color = "b"
        else:
            color = "g"

        # Create a Rectangle patch
        rect = patches.Rectangle(
            (xmin, ymin), (xmax - xmin), (ymax - ymin), linewidth=2, edgecolor=color, facecolor="none"
        )

        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()