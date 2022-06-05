from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.callbacks.ssl_online import SSLOnlineEvaluator
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC
from torchmetrics.functional import accuracy


class ExactOnlineEval(SSLOnlineEvaluator):
    """This class is mostly copy of its parent class with some small changes.
        the model that uses this callback requires:

        Requirements:
            - pl_module.all_val_online_logits which is list of all val logits for the whole epoch
            - pl_module.all_test_online_logits
            - datamodule should have val_ds
            - val_ds should have list of all labels in val_ds.labels


    """

    def __init__(self, *args, **kwargs):
        """
        Args:
            z_dim: Representation dimension
            drop_p: Dropout probability
            hidden_dim: Hidden dimension for the fine-tune MLP
        """
        super().__init__(*args, **kwargs)

        self.num_classes = kwargs['num_classes']
        self.setup_flag = True

        # # metrics for logging
        metrics = MetricCollection({
            'online_acc-macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
            # 'finetune_auc': AUROC(num_classes=self.num_classes),
        })
        self.train_acc = Accuracy()
        self.val_metrics = metrics.clone(prefix='val/ssl/')
        self.test_metrics = metrics.clone(prefix='test/ssl/')

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        super(ExactOnlineEval, self).setup(trainer=trainer, pl_module=pl_module, stage=stage)

        if self.setup_flag:
            pl_module.train_acc = self.train_acc.to(pl_module.device)
            pl_module.val_metrics = self.val_metrics.to(pl_module.device)
            pl_module.test_metrics = self.test_metrics.to(pl_module.device)
            self.setup_flag = False


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

        # acc = accuracy(mlp_logits.softmax(-1), y)

        return mlp_loss, mlp_logits, y

    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mlp_loss, mlp_logits, y = self.shared_step(pl_module, batch)

        # update finetune weights
        mlp_loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        pl_module.train_acc(mlp_logits.softmax(-1), y)

        pl_module.log("train/ssl/online_loss", mlp_loss, on_step=False, on_epoch=True, sync_dist=True)
        pl_module.log("train/ssl/online_acc", pl_module.train_acc.computed(), on_step=False, on_epoch=True,
                      sync_dist=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        mlp_loss, mlp_logits, y = self.shared_step(pl_module, batch)
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}

        if dataloader_idx == 0:
            pl_module.all_val_online_logits.append(mlp_logits)
            pl_module.val_metrics(mlp_logits.softmax(-1), y)

            pl_module.log_dict(pl_module.val_metrics, **kwargs)
            pl_module.log("val/ssl/online_loss", mlp_loss, **kwargs)
            # pl_module.log("val/ssl/online_acc", val_acc, on_step=False, on_epoch=True, sync_dist=True)
        elif dataloader_idx == 1:
            pl_module.all_test_online_logits.append(mlp_logits)
            pl_module.test_metrics(mlp_logits.softmax(-1), y)

            pl_module.log_dict(pl_module.test_metrics, **kwargs)
            pl_module.log("test/ssl/online_loss", mlp_loss, **kwargs)


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reset metrics
        pl_module.train_acc.reset()
        pl_module.val_metrics.reset()
        pl_module.test_metrics.reset()


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
