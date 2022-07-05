from contextlib import contextmanager
from typing import Any, Dict, Optional, Sequence, Tuple, Union
from pytorch_lightning.utilities.types import STEP_OUTPUT

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

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

        # # metrics for logging
        # self.metrics = MetricCollection({
        #     'online_acc_macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
        #     # 'finetune_auc': AUROC(num_classes=self.num_classes),
        # })
        # self.all_val_test_mlp_logits = {}
        #
        # self.all_centers_val_logits = []
        # self.all_centers_val_labels = []
        # self.val_macroLoss_all_centers = []
        #
        # self.all_centers_test_logits = []
        # self.all_centers_test_labels = []
        # self.test_macroLoss_all_centers = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        super(ExactOnlineEval, self).setup(trainer=trainer, pl_module=pl_module, stage=stage)

        if self.setup_flag:
            self.setup_flag = False

            pl_module.train_acc = Accuracy().to(pl_module.device)
            # pl_module.val_metrics_centers_dict = {}
            # pl_module.test_metrics_centers_dict = {}

            # if isinstance(trainer.datamodule.val_ds, dict) and isinstance(trainer.datamodule.test_ds, dict):
            #     # defining metrics for each center individually
            #     for key in trainer.datamodule.val_ds.keys():
            #         prefix = 'val/ssl/'+key+'/'
            #         pl_module.val_metrics_centers_dict[key] = self.metrics.clone(prefix=prefix).to(pl_module.device)
            #     for key in trainer.datamodule.test_ds.keys():
            #         prefix = 'test/ssl/'+key+'/'
            #         pl_module.test_metrics_centers_dict[key] = self.metrics.clone(prefix=prefix).to(pl_module.device)
            #
            # elif isinstance(trainer.datamodule.val_ds, dict) or isinstance(trainer.datamodule.test_ds, dict):
            #     raise ValueError("both val_ds and test_ds should use the same centers")
            #
            # else:
            #     key = trainer.datamodule.cohort_specifier
            #     prefix = 'val/ssl/' + key + '/'
            #     pl_module.val_metrics_centers_dict[key] = self.metrics.clone(prefix=prefix).to(pl_module.device)
            #
            #     prefix = 'test/ssl/' + key + '/'
            #     pl_module.test_metrics_centers_dict[key] = self.metrics.clone(prefix=prefix).to(pl_module.device)
            #
            # # defining metric for combination of centers
            # pl_module.val_microMetrics_all_centers = self.metrics.clone(prefix='val/ssl/').to(pl_module.device)
            # pl_module.test_microMetrics_all_centers = self.metrics.clone(prefix='test/ssl/').to(pl_module.device)

    def shared_step(self, pl_module: LightningModule, batch: Sequence):
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
        outputs: Optional[STEP_OUTPUT],
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
        pl_module.log("train/ssl/online_acc", pl_module.train_acc.compute(), on_step=False, on_epoch=True,
                      sync_dist=True)

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Optional[STEP_OUTPUT],
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        # mlp feed forward
        mlp_loss, mlp_logits, y = self.shared_step(pl_module, batch)

        self.logging_combined_centers_loss(
            pl_module,
            trainer,
            dataloader_idx,
            mlp_loss,
        )

        # reassigning the output. This will be accessed in metric_logger ## todo check if it works
        outputs[1] = mlp_logits
        outputs[2] = y

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        pass

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reset metrics
        pl_module.train_acc.reset()

        # for value in pl_module.val_metrics_centers_dict.values():
        #     value.reset()
        # for value in pl_module.test_metrics_centers_dict.values():
        #     value.reset()
        #
        # pl_module.val_microMetrics_all_centers.reset()
        # pl_module.test_microMetrics_all_centers.reset()
        #
        # self.all_centers_val_logits = []
        # self.all_centers_val_labels = []
        # self.val_macroLoss_all_centers = []
        #
        # self.all_centers_test_logits = []
        # self.all_centers_test_labels = []
        # self.test_macroLoss_all_centers = []

    def logging_combined_centers_loss(
            self,
            pl_module,
            trainer,
            dataloader_idx,
            mlp_loss,
    ):
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}

        if dataloader_idx < len(pl_module.val_metrics_centers_dict):
            # logit_key = pl_module.val_metrics_centers_dict.keys()[dataloader_idx]
            # self.all_val_test_mlp_logits['val_'+logit_key]

            # val_metric_cur_center = pl_module.val_metrics_centers_dict.values()[dataloader_idx]
            #
            # # computing metric for the center
            # pl_module.val_metric_cur_center(mlp_logits.softmax(-1), y)
            # pl_module.log_dict(val_metric_cur_center, **kwargs)
            #
            # # logging logits and lables for all centers for micro acc
            # self.all_centers_val_logits.append(mlp_logits)
            # self.all_centers_val_labels.append(y)
            self.val_macroLoss_all_centers.append(mlp_loss)

            if dataloader_idx == len(pl_module.val_metrics_centers_dict) - 1:
                # logits = torch.cat(self.all_centers_val_logits)
                # labels = torch.cat(self.all_centers_val_labels)
                # pl_module.val_microMetrics_all_centers(logits.softmax(-1), labels)
                # pl_module.log_dict(pl_module.val_microMetrics_all_centers, **kwargs)
                mlp_losses = torch.mean(self.val_macroLoss_all_centers)
                pl_module.log("val/ssl/online_loss", mlp_losses, **kwargs)

        else:
            # idx = dataloader_idx - len(pl_module.val_metrics_centers_dict)
            # test_metric_cur_center = pl_module.test_metrics_centers_dict.values()[idx]
            #
            # # computing metric for the center
            # pl_module.test_metric_cur_center(mlp_logits.softmax(-1), y)
            # pl_module.log_dict(test_metric_cur_center, **kwargs)
            #
            # # logging logits and lables for all centers for micro acc
            # self.all_centers_test_logits.append(mlp_logits)
            # self.all_centers_test_labels.append(y)
            self.test_macroLoss_all_centers.append(mlp_loss)

            if dataloader_idx == len(pl_module.val_metrics_centers_dict) + len(pl_module.test_metrics_centers_dict) - 1:
                # logits = torch.cat(self.all_centers_test_logits)
                # labels = torch.cat(self.all_centers_test_labels)
                # pl_module.test_microMetrics_all_centers(logits.softmax(-1), labels)
                # pl_module.log_dict(pl_module.test_microMetrics_all_centers, **kwargs)
                mlp_losses = torch.mean(self.test_macroLoss_all_centers)
                pl_module.log("test/ssl/online_loss", mlp_losses, **kwargs)

            # pl_module.all_test_online_logits.append(mlp_logits)
            # pl_module.test_metrics(mlp_logits.softmax(-1), y)
            #
            # pl_module.log_dict(pl_module.test_metrics, **kwargs)
            # pl_module.log("test/ssl/online_loss", mlp_loss, **kwargs)


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
