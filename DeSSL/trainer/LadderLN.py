from typing import Tuple, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.core import LightningModule


class Ladder(LightningModule):
    def __init__(self,
                 train_and_val_loader: Tuple[DataLoader, DataLoader],
                 optimizer: dict,
                 lr_schedular: dict,
                 model: nn.Module,
                 lam_list: List[float]):
        super().__init__()
        self._loader = train_and_val_loader
        self._optimizer = optimizer
        self._lr_schedular = lr_schedular
        self.model = model
        self.lam_list = lam_list
        # self.save_hyperparameters()

    def forward(self, path_name, *input):
        return self.model(path_name, *input)

    def training_step(self, batch, batch_idx):
        (label_data, target), (unlabel_data, _) = batch
        self('clear', label_data, unlabel_data)
        output = self('noise', label_data, unlabel_data)
        self('decoder')
        loss = self.model.get_loss_d(self.lam_list)
        loss_train = loss + F.nll_loss(torch.log(output), target)
        self.log('train/loss', loss_train, on_step=True,
                 on_epoch=True, logger=True)
        return loss_train

    def validation_step(self, batch, batch_idx):
        input, target = batch
        output = self('clear', input)
        loss_val = F.nll_loss(torch.log(output), target)
        acc1, acc5 = self.__accuracy(output, target, topk=(1, 5))
        self.log('val_loss', loss_val, on_epoch=True)
        self.log('val_acc1', acc1, prog_bar=True, on_epoch=True)
        self.log('val_acc5', acc5, on_epoch=True)

    @staticmethod
    def __accuracy(output, target, topk=(1, )):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = correct[:k].reshape(-1).float().sum(0,
                                                                keepdim=True)
                res.append(correct_k.mul_(100.0 / batch_size))
            return res

    def configure_optimizers(self):
        optimizer_fn = self._optimizer['optimizer']
        kwargs = {}
        for k, v in self._optimizer.items():
            if k != 'optimizer':
                kwargs[k] = v
        optimizer = optimizer_fn(self.parameters(), **kwargs)
        lr_scheduler_fn = self._lr_schedular['lr_scheduler']
        kwargs = {}
        for k, v in self._lr_schedular.items():
            if k != 'lr_scheduler':
                kwargs[k] = v
        lr_scheduler = lr_scheduler_fn(optimizer, **kwargs)
        return [optimizer], [lr_scheduler]

    def train_dataloader(self):
        return self._loader[0]

    def val_dataloader(self):
        return self._loader[1]
