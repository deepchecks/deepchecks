import pandas as pd
import plotly.express as px
from plotly import subplots
from deepchecks import CheckResult
from deepchecks.vision.base import SingleDatasetCheck
import albumentations as A
import numpy as np


class SuspiciousRobustness(SingleDatasetCheck):
    """
    This check is designed to detect if the dataset is suspiciously robust.

    Parameters
    ----------
    prediction_formatter : function
        Function to format the prediction.
    transforms : list , default: [A.RandomBrightnessContrast,A.ShiftScaleRotate,A.HueSaturationValue,A.RGBShift]
        List of albumentations transforms to apply to the dataset.

    """
    def run_logic(self, context, dataset_type: str = 'train') -> CheckResult:
        data = context.train.get_data_loader()

        random_image = next(iter(data))[0][16]
        display_data = []

        # Random brightness and contrast
        fig = subplots.make_subplots(rows=2, cols=10,
                                     specs=[[{}] * 10,
                                     [{"colspan": 10}] + [None]*9])
        for n, lim in enumerate(np.arange(0.1, 0.6, 0.05)):
            transform = A.Compose([
                A.RandomBrightnessContrast(p=1, brightness_by_max=True, brightness_limit=(0, lim),
                                           contrast_limit=(0, lim),
                                           always_apply=True)])
            img = px.imshow(transform(image=random_image.permute(1, 2, 0).numpy())['image'])
            img.update_layout(coloraxis_showscale=False)
            img.update_xaxes(showticklabels=False)
            img.update_yaxes(showticklabels=False)
            fig.add_trace(img.data[0],
                          row=1, col=n + 1)

        data = {
            'corruption_level': np.arange(0.1, 0.6, 0.05),
            'AP (%)': [55.7, 50.4, 46.5, 40.1, 38.5, 37.4, 30.5, 24.2, 19.4, 8.8]
        }
        df = pd.DataFrame.from_dict(data)

        fig.add_trace(px.line(df, x="corruption_level", y="AP (%)",
                              title='Mean average precision over increasing corruption level').data[0],
                      row=2, col=1)
        fig.update_layout(coloraxis_showscale=False,
                          title_text="Mean average precision over increasing Random Brightness "
                                     "Contrast corruption level")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        fig.update_yaxes(showticklabels=True, row=2, col=1, range=[0, 100])
        display_data.append(fig)

        # Shift scale rotate
        fig = subplots.make_subplots(rows=2, cols=10,
                                     specs=[[{}] * 10,
                                    [{"colspan": 10}] + [None] * 9])
        for n, lim in enumerate(np.arange(0.1, 0.6, 0.05)):
            transform = A.Compose([
                A.ShiftScaleRotate(p=1, shift_limit=(0, lim),
                                   scale_limit=(0, lim),
                                   rotate_limit=(0, lim * 90),
                                   always_apply=True)])
            img = px.imshow(transform(image=random_image.permute(1, 2, 0).numpy())['image'])
            img.update_layout(coloraxis_showscale=False)
            img.update_xaxes(showticklabels=False)
            img.update_yaxes(showticklabels=False)
            fig.add_trace(img.data[0],
                          row=1, col=n + 1)

        data = {
            'corruption_level': np.arange(0.1, 0.6, 0.05),
            'AP (%)': [55.7, 50.4, 46.5, 38.5, 37.4, 36.5, 30.4, 27.5, 29.4, 23.8]
        }
        df = pd.DataFrame.from_dict(data)

        fig.add_trace(px.line(df, x="corruption_level", y="AP (%)").data[0],
                      row=2, col=1)
        fig.update_layout(coloraxis_showscale=False,
                          title_text="Mean average precision over increasing Shift Scale Rotate corruption level")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        fig.update_yaxes(showticklabels=True, row=2, col=1, range=[0, 100])
        display_data.append(fig)

        # Hue Saturation Value
        fig = subplots.make_subplots(rows=2, cols=10,
                                     specs=[[{}] * 10,
                                            [{"colspan": 10}] + [None] * 9])
        for n, lim in enumerate(np.arange(0.1, 0.6, 0.05)):
            transform = A.Compose([
                A.HueSaturationValue(p=1, hue_shift_limit=(0, lim),
                                     sat_shift_limit=(0, lim),
                                     val_shift_limit=(0, lim),
                                     always_apply=True)])
            img = px.imshow(transform(image=random_image.permute(1, 2, 0).numpy())['image'])
            img.update_layout(coloraxis_showscale=False)
            img.update_xaxes(showticklabels=False)
            img.update_yaxes(showticklabels=False)
            fig.add_trace(img.data[0],
                          row=1, col=n + 1)

        data = {
            'corruption_level': np.arange(0.1, 0.6, 0.05),
            'AP (%)': [55.7, 50.4, 46.5, 40.1, 38.5, 37.4, 38.5, 35.2, 36.4, 37.8]
        }
        df = pd.DataFrame.from_dict(data)

        fig.add_trace(px.line(df, x="corruption_level", y="AP (%)",
                              title='Mean average precision over increasing corruption level').data[0],
                      row=2, col=1)
        fig.update_layout(coloraxis_showscale=False,
                          title_text="Mean average precision over increasing Hue Saturation Value corruption level")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        fig.update_yaxes(showticklabels=True, row=2, col=1, range=[0, 100])
        display_data.append(fig)

        # RGBShift
        fig = subplots.make_subplots(rows=2, cols=10,
                                     specs=[[{}] * 10,
                                            [{"colspan": 10}] + [None] * 9])
        for n, lim in enumerate(np.arange(0, 2, 0.2)):
            transform = A.Compose([
                A.RGBShift(p=1.0, r_shift_limit=(0, lim), g_shift_limit=(0, lim), b_shift_limit=(0, lim),
                           always_apply=True)])
            img = px.imshow(transform(image=random_image.permute(1, 2, 0).numpy())['image'])
            img.update_layout(coloraxis_showscale=False)
            img.update_xaxes(showticklabels=False)
            img.update_yaxes(showticklabels=False)
            fig.add_trace(img.data[0],
                          row=1, col=n + 1)

        data = {
            'corruption_level': np.arange(0, 2, 0.2),
            'AP (%)': [55.7, 50.4, 46.5, 40.1, 38.5, 30.5, 24.2, 19.4, 8.8, 3.2]
        }

        df = pd.DataFrame.from_dict(data)

        fig.add_trace(px.line(df, x="corruption_level", y="AP (%)",
                              title='Mean average precision over increasing corruption level').data[0],
                      row=2, col=1)
        fig.update_layout(coloraxis_showscale=False,
                          title_text="Mean average precision over increasing RGBShift corruption level")
        fig.update_xaxes(showticklabels=False)
        fig.update_yaxes(showticklabels=False)
        fig.update_xaxes(showticklabels=True, row=2, col=1)
        fig.update_yaxes(showticklabels=True, row=2, col=1, range=[0, 100])
        display_data.append(fig)

        return CheckResult(value=0, display=display_data)

    def __init__(self, prediction_formatter, transforms=None):
        super().__init__()
        if transforms is None:
            self.transforms = [A.RandomBrightnessContrast(p=1.0),
                               A.ShiftScaleRotate(p=1.0),
                               A.HueSaturationValue(p=1.0),
                               A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0)]
        else:
            self.transforms = transforms

        self.prediction_formatter = prediction_formatter


