from ignite.metrics import Metric
from ignite.metrics.metric import sync_all_reduce, reinit__is_reduced
import torch
from torch.nn import functional as F


class Accuracy(Metric):

    def __init__(self, num_classes, *args, **kwargs):
        self._num_correct = [0] * num_classes
        self._num_examples = [0] * num_classes
        self._num_classes = num_classes
        super().__init__(*args, **kwargs)
        self.i = 0

    @reinit__is_reduced
    def reset(self):
        self._num_correct = [0] * self._num_classes
        self._num_examples = [0] * self._num_classes
        self.i = 0
        super(Accuracy, self).reset()


    @reinit__is_reduced
    def update(self, output):
        y_pred, y = output

        for predtensor,real in zip(y_pred,y):
            pred = torch.argmax(F.softmax(predtensor,0), 0)
            self._num_examples[real] += 1
            if real == pred:
                self._num_correct[real] += 1

    @sync_all_reduce("_num_examples", "_num_correct")
    def compute(self):
        class_accuracy = torch.tensor([0.0] * self._num_classes)
        for i, (correct, total) in enumerate(zip(self._num_correct, self._num_examples)):
            class_accuracy[i] = correct / total
        #if self._num_examples == 0:
        #    raise NotComputableError('AccuracyPerClass must have at least one example before it can be computed.')
        return class_accuracy