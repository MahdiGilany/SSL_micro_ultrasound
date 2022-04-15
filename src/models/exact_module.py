from typing import Any, List

import torch
import torch_optimizer as trch_opt
import wandb
from pytorch_lightning import LightningModule
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

from src.models.components.simple_dense_net import SimpleDenseNet


class ExactLitModule(LightningModule):
    """Example of LightningModule for MNIST classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        lr: float = 0.001,
        weight_decay: float = 0.0005,
        batch_size: int = 32,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # macro means balanced accuracy
        self.train_acc = Accuracy() #average='macro', num_classes=2)
        self.val_acc = Accuracy() #average='macro', num_classes=2)
        self.test_acc = Accuracy() #average='macro', num_classes=2)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

    def forward(self, x: torch.Tensor):
        return self.net(x)

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log train metrics
        acc = self.train_acc(preds, targets)
        lr = self.onecyc_scheduler.get_last_lr()

        self.log("lr", lr[0], on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in `training_epoch_end()`` below
        # remember to always return loss from `training_step()` or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def on_before_zero_grad(self, optimizer) -> None:
        # for OneCycLR we need to take a step for each batch
        self.onecyc_scheduler.step()

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx):
        loss, preds, targets = self.step(batch)

        if dataloader_idx == 0:
            # log val metrics
            acc = self.val_acc(preds, targets)
            self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
            self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        elif dataloader_idx == 1:
            # log test metrics
            acc = self.test_acc(preds, targets)
            self.log("test/loss", loss, on_step=False, on_epoch=True)
            self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs: List[Any]):
        acc = self.val_acc.compute()  # get val accuracy from current epoch
        self.val_acc_best.update(acc)
        self.log("val/acc_best/acc_best", self.val_acc_best.compute(), on_epoch=True, prog_bar=True)

        # logging confusion matrix
        preds = torch.cat([i['preds'] for i in outputs[0]], dim=0).detach().cpu().numpy()
        targets = torch.cat([i['targets'] for i in outputs[0]], dim=0).detach().cpu().numpy()
        wandb_conf = wandb.plot.confusion_matrix(probs=None, y_true=targets, preds=preds, class_names=["0", "1"])
        self.logger.experiment.log({"val/conf_mat": wandb_conf})
        #
        # acc2 = (preds == targets)
        # acc2 = acc2.sum()/len(acc2)
        # self.log('val/acc2', acc2, on_epoch=True)

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log test metrics
        acc = self.test_acc(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_epoch_end(self, outputs: List[Any]):
        pass

    def on_epoch_end(self):
        # reset metrics at the end of every epoch
        self.train_acc.reset()
        self.test_acc.reset()
        self.val_acc.reset()

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """

        n_epochs = self.trainer.max_epochs
        len_ds = len(self.trainer.datamodule.train_ds)
        steps_per_epoch = int(len_ds / self.hparams.batch_size)
        print(len_ds, steps_per_epoch)

        # self.optimizer = torch.optim.Adam(params=self.parameters(), lr=self.hparams.lr,
        #                                   weight_decay=self.hparams.weight_decay)

        self.optimizer = trch_opt.NovoGrad(self.parameters(), lr=float(self.hparams.lr),
                                           weight_decay=self.hparams.weight_decay)

        self.onecyc_scheduler = \
            torch.optim.lr_scheduler.OneCycleLR(self.optimizer, float(self.hparams.lr), epochs=n_epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95, div_factor=10.0,
                                                final_div_factor=10000.0, three_phase=False,
                                                last_epoch=-1, verbose=False)

        return self.optimizer #, [self.onecyc_scheduler] removed since needs steps for each batch
