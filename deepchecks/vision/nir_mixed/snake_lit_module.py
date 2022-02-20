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
                 optimizer: str = "adam",
                 finetune_last: bool = True,
                 lr: float = 1e-3,
                 *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        optimizers = {'adam': Adam, 'sgd': SGD}
        self.lr = lr
        self.optimizer = optimizers[optimizer.lower()]
        self.num_classes = num_classes

        # init a pretrained resnet
        backbone = models.resnet34(pretrained=True)
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)

        # fine tune the fully-connected layers rather than train everything
        if finetune_last:
            for child in list(self.feature_extractor.children())[:-1]:
                for param in child.parameters():
                    param.requires_grad = False

        # set loss and criterion
        num_filters = backbone.fc.in_features
        linear_size = list(backbone.children())[-1].in_features
        assert linear_size == num_filters
        self.classifier = nn.Linear(num_filters, num_classes)
        self.criterion = nn.CrossEntropyLoss()
        # Important for loading
        self.save_hyperparameters()

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

    def configure_optimizers(self):
        return self.optimizer(self.parameters(), lr=self.lr)

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
        loss, acc = self._step(x, y)
        # perform logging
        self.log("test_acc", acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log("test_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss