from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC


class MetricLogger(Callback):
    """Computes and logs all patch-wise metrics
        the model that uses this callback requires:

        Requirements:
            - considers that pl_module.all_val_online_logits variable is available and contarins all logits
            - considers that pl_module.all_test_online_logits variable is available and contarins all logits
            - considers that pl_module.val_metrics['finetune(or)online_acc-macro'] variable is available
            - considers that pl_module.test_metrics['finetune(or)online_acc-macro'] variable is available
            - trainer.datamodule.val_ds.labels
            - trainer.datamodule.test_ds.labels
    """

    def __init__(
            self,
            mode: Literal["online", "finetune"] = "finetune",
            num_classes: int = 2,
    ):

        super().__init__()

        self.mode = mode
        self.num_classes = num_classes
        self.kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True}

        # metrics for logging
        metrics = MetricCollection({
            # f'{mode}_acc': Accuracy(num_classes=self.num_classes, multiclass=True),
            # f'{mode}_acc-macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
            f'{mode}_auc': AUROC(num_classes=self.num_classes),
        })

        self.val_prefix = "val/" if mode == "finetune" else "val/ssl/"
        self.val_metrics = metrics.clone(prefix=self.val_prefix)
        self.val_cf = ConfusionMatrix(num_classes=self.num_classes)
        self.val_acc_best = MaxMetric()
        self.val_auc_best = MaxMetric()

        self.test_prefix = "test/" if mode == "finetune" else "test/ssl/"
        self.test_metrics = metrics.clone(prefix=self.test_prefix)
        self.test_cf = ConfusionMatrix(num_classes=self.num_classes)
        self.test_acc_best = MaxMetric()
        self.test_auc_best = MaxMetric()

    def on_validation_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Sequence,
        batch: Sequence,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        pass
        # _, y = batch
        # y = y.detach().cpu()
        #
        # if dataloader_idx == 0:
        #
        #
        # elif dataloader_idx == 1:
        #     logits = pl_module.all_test_online_logits[-1]
        #     logits = logits.detach().cpu()
        #     self.test_metrics(logits.softmax(-1), y)
        #     self.test_cf(logits.softmax(-1), y)
        #     pl_module.log_dict(self.test_metrics.compute(), prog_bar=True, **self.kwargs)


    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ### validation logs ###
        logits = torch.cat(pl_module.all_val_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.val_ds.labels)[:len(logits)]

        self.val_metrics(logits.softmax(-1), labels) #auc for now
        self.val_cf(logits.softmax(-1), labels)
        cf = self.val_cf.compute()
        tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        pl_module.log_dict(self.val_metrics.compute(), prog_bar=True, **self.kwargs)
        pl_module.log(f"{self.val_prefix}{self.mode}_sen", tp / (tp + fn), **self.kwargs)
        pl_module.log(f"{self.val_prefix}{self.mode}_spe", tn / (tn + fp), **self.kwargs)

        # maximum acc and auc
        val_auc = self.val_metrics[f'{self.mode}_auc'].compute()
        val_acc = pl_module.val_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()

        self.val_auc_best.update(val_auc)
        self.val_acc_best.update(val_acc)

        pl_module.log(f"{self.val_prefix}{self.mode}_auc_best", self.val_auc_best.compute(), **self.kwargs)
        pl_module.log(f"{self.val_prefix}{self.mode}_acc-macro_best", self.val_acc_best.compute(), prog_bar=True,
                      **self.kwargs)


        ### test logs ###
        logits = torch.cat(pl_module.all_test_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.test_ds.labels)[:len(logits)]

        self.test_metrics(logits.softmax(-1), labels)
        self.test_cf(logits.softmax(-1), labels)
        cf = self.test_cf.compute()
        tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        pl_module.log_dict(self.test_metrics.compute(), prog_bar=True, **self.kwargs)
        pl_module.log(f"{self.test_prefix}{self.mode}_sen", tp / (tp + fn), **self.kwargs)
        pl_module.log(f"{self.test_prefix}{self.mode}_spe", tn / (tn + fp), **self.kwargs)

        # maximum acc and auc
        test_auc = self.test_metrics[f'{self.mode}_auc'].compute()
        test_acc = pl_module.test_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()

        self.test_auc_best.update(test_auc)
        self.test_acc_best.update(test_acc)

        pl_module.log(f"{self.test_prefix}{self.mode}_auc_best", self.test_auc_best.compute(), **self.kwargs)
        pl_module.log(f"{self.test_prefix}{self.mode}_acc-macro_best", self.test_acc_best.compute(), prog_bar=True,
                      **self.kwargs)

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.val_metrics.reset()
        self.test_metrics.reset()
        self.val_cf.reset()
        self.test_cf.reset()

        # reset metrics after sanity checks'
        if trainer.sanity_checking:
            self.val_acc_best.reset()
            self.val_auc_best.reset()
            self.test_acc_best.reset()
            self.test_auc_best.reset()
