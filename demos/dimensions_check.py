import pandas as pd
import plotly.figure_factory as ff
import numpy as np
from deepchecks import CheckResult
from deepchecks.vision.base import SingleDatasetCheck


class ImageDimensionsCheck(SingleDatasetCheck):

    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        vis_dataset = context.train if dataset_type == 'train' else context.test

        dataset = vis_dataset.get_data_loader().dataset
        raw_data = []
        for img, annotation in dataset:
            raw_data.append([img.shape, annotation])

        im_w_list = []
        im_h_list = []
        num_boxes = []
        labels = []
        for img_shape, annotation in raw_data:
            im_w_list.append(img_shape[2])
            im_h_list.append(img_shape[1])
            num_boxes.append(annotation["boxes"].shape[0])
            labels += annotation["labels"].tolist()

        im_w_list_fig = ff.create_distplot([im_w_list], ['width'], bin_size=10)
        im_w_list_fig.update_layout(title='Image width distribution')

        im_h_list_fig = ff.create_distplot([im_h_list], ['height'], bin_size=10)
        im_h_list_fig.update_layout(title='Image height distribution')

        num_boxes_fig = ff.create_distplot([num_boxes], ['num_boxes'], bin_size=5)
        num_boxes_fig.update_layout(title='Number of boxes distribution')

        labels = np.array(labels)

        # in 603- we have 115 masks
        # in 15- we have a small image (>3sigma)
        # print(f"Mean image width: {np.mean(im_w_list)}; Image width std: {np.std(im_w_list)}")
        # print(f"Mean image height: {np.mean(im_h_list)}; Image height std: {np.std(im_h_list)}")
        # print(f"Mean num of bbox: {np.mean(num_boxes)}; Num of bbox std: {np.std(num_boxes)=}")
        # hist, _ = np.histogram(labels, bins=3)
        # print(f"{hist=}")

        return CheckResult(value=0, display=[im_w_list_fig, im_h_list_fig, num_boxes_fig])

    def __init__(self):
        super().__init__()
