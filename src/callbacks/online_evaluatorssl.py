from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import MaxMetric
from torchmetrics.functional import accuracy


class ExactOnlineEval(SSLOnlineEvaluator):
    """This class is mostly copy of its parent class with some small changes."""

    def __init__(self, *args, **kwargs):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__(*args, **kwargs)

        # for logging best so far validation accuracy
        self.val_online_acc_best = MaxMetric()

    def shared_step(
        self,
        pl_module: LightningModule,
        batch: Sequence,
    ):
        with torch.no_grad():
            with set_training(pl_module, False):
                x, y = self.to_device(batch, pl_module.device) # todo only one linear layer added on top of one branch
                representations = (pl_module(x)["feats"]).flatten(start_dim=1)

        # forward pass
        mlp_logits = self.online_evaluator(representations)  # type: ignore[operator]
        mlp_loss = F.cross_entropy(mlp_logits, y)

        acc = accuracy(mlp_logits.softmax(-1), y)

        return acc, mlp_loss, mlp_logits

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        train_acc, mlp_loss, _ = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.log("train/ssl/online_acc", train_acc, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/ssl/online_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        val_acc, mlp_loss, mlp_logits = self.shared_step(pl_module, batch)

        if dataloader_idx == 0:
            pl_module.all_val_online_logits.append(mlp_logits)
            pl_module.log("val/ssl/online_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("val/ssl/online_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        elif dataloader_idx == 1:
            pl_module.all_test_online_logits.append(mlp_logits)
            pl_module.log("test/ssl/online_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
            pl_module.log("test/ssl/online_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        # saving all preds for corewise callback: val and test
        all_val_online_preds = torch.cat(pl_module.all_val_online_logits).argmax(dim=1).detach().cpu().numpy()
        # all_test_online_preds = torch.cat(self.all_test_online_logits.argmax(dim=1), dim=0).detach().cpu().numpy()

        # logging the best val online_acc
        val_targets = trainer.datamodule.val_ds.labels[:len(all_val_online_preds)]
        val_acc = (all_val_online_preds == val_targets).sum() / len(val_targets)
        self.val_online_acc_best.update(val_acc)
        pl_module.log("val/ssl/online_acc_best", self.val_online_acc_best.compute(), on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reset metrics after sanity checks'
        if trainer.sanity_checking:
            self.val_online_acc_best.reset()



@contextmanager
def set_training(module: nn.Module, mode: bool):
    """Context manager to set training mode.

    When exit, recover the original training mode.
    Args:
        module: module to set training mode
        mode: whether to set training mode (True) or evaluation mode (False).
    """
    original_mode = module.training

    try:
        module.train(mode)
        yield module
    finally:
        module.train(original_mode)
