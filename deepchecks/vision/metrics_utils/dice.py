from collections import defaultdict
from typing import Tuple

import numpy as np
import torch
from ignite.metrics import Metric


class MeanDice(Metric):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})
        # self._class_ids = []
        # self._tp = []
        # self._total_y_pred = []
        # self._total_y = []

    def reset(self) -> None:
        super().reset()
        self._evals = defaultdict(lambda: {'dice': 0, 'count': 0})
        self._i = 0
        # self._class_ids = []
        # self._tp = []
        # self._total_y_pred = []
        # self._total_y = []

    def update(self, output: Tuple[torch.Tensor, torch.Tensor]):
        y_pred, y = output

        for i in range(len(y)):
            for class_id in [int(x) for x in torch.unique(y[i])]:
                y_pred_this_label = (y_pred[i] == class_id).numpy()
                y_this_label = (y[i] == class_id).numpy()
                # self._class_ids.append(class_id)
                # self._tp.append(np.logical_and(y_pred_this_label, y_this_label).sum())
                # self._total_y_pred.append(y_pred_this_label.sum())
                # self._total_y.append(y_this_label.sum())

                tp = np.logical_and(y_pred_this_label, y_this_label).sum()
                total_y_pred = y_pred_this_label.sum()
                total_y = y_this_label.sum()

                self._evals[class_id]['dice'] += (2*tp) / (total_y_pred + total_y)
                self._evals[class_id]['count'] += 1
                self._i += 1

    def compute(self):
        sorted_classes = [int(class_id) for class_id in sorted(self._evals.keys())]
        ret = []
        for class_id in sorted_classes:
            count = self._evals[class_id]['count']
            dice = self._evals[class_id]['dice']
            mean_dice = dice / count if count != 0 else 0
            ret.append(mean_dice)
        return ret
