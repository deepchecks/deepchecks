import sys
from abc import ABC
from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from torch.optim import SGD, Adam


class SnakeLitModule(pl.LightningModule, ABC):
    def __init__(self,
                 num_classes: int,
                 optimizer: str = "sgd",
                 finetune_last: bool = True,
                 lr: float = 1e-3,
                 debug: bool = False,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.lr = lr
        self.optimizer = optimizers[optimizer.lower()]
        self.num_classes = num_classes

        # init a pretrained resnet, keep N-1 layers
        backbone = models.resnet34(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # set grad computation for all layers of feature extractor to false
        if finetune_last:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False

        # Debug mode keeps the ResNet classifier, used it to verify model loading properly
        if debug:
            self.classifier = list(backbone.children())[-1]
            print("Warning: Using ResNet as-is, are you sure that's what you want?",
                  file=sys.stderr)
        # Otherwise we freeze layers and
        else:
            # create new FC layer
            num_filters = backbone.fc.in_features
            self.classifier = nn.Linear(num_filters, num_classes)

        # set loss and criterion
        self.criterion = nn.CrossEntropyLoss()
        # Important for loading
        self.save_hyperparameters()

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

    def _step(self, x, y):
        """
        https://scikit-learn.org/stable/modules/model_evaluation.html#regression-metrics
        :param x:
        :param y:
        :return:
        """
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        accuracy = (y == torch.argmax(y_hat, 1)).type(torch.FloatTensor).mean()
        return loss, accuracy

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, acc = self._step(x, y)
        # perform logging
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        loss, acc = self._step(x, y)
        # perform logging
        self.log("val_loss", loss, on_epoch=True, prog_bar=True, logger=True)
        self.log("val_acc", acc, on_epoch=True, prog_bar=True, logger=True)

    def forward(self, x):
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def predict_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        """
        If the input contains more than just the image, we disregard it (prediction doesn't expect labeled data)
        :param batch:
        :param batch_idx:
        :param dataloader_idx:
        :return:
        """
        if len(batch) > 1:
            x = batch[0]
        else:
            x = batch
        y_hat = self(x)
        out = torch.argmax(y_hat, 1)
        return out

    def test_step(self, batch, batch_idx: int , dataloader_idx: int = None):
        x, y = batch
        y_hat = self(x)
        acc = (y == torch.argmax(y_hat, 1)).type(torch.FloatTensor).mean()
        _, tk = torch.topk(y_hat, 5, dim=1)
        top5 = tk.t().eq(y.view(1, -1).expand_as(tk.t()))
        a, _ = top5.max(0)
        final_top5 = a.float().mean()
        # perform logging
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_top5", final_top5, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return acc