from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC, CatMetric
from torchmetrics.functional import accuracy, confusion_matrix, auroc


class MetricLogger(Callback):
    """Computes and logs all patch-wise metrics
        the model that uses this callback requires:

        # todo: all information that is assumed to be available can be obtained in on_validation_batch_end...
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
            corewise_inv_threshold: float = 0.5,
    ):

        super().__init__()

        self.mode = mode
        self.num_classes = num_classes
        self.inv_threshold = corewise_inv_threshold
        self.kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True}

        # metrics for logging
        # metrics = MetricCollection({
        #     # f'{mode}_acc': Accuracy(num_classes=self.num_classes, multiclass=True),
        #     # f'{mode}_acc-macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
        #     f'{mode}_auc': AUROC(num_classes=self.num_classes),
        # })

        self.val_prefix = "val/" if mode == "finetune" else "val/ssl/"
        # self.val_metrics = metrics.clone(prefix=self.val_prefix)
        # self.val_cf = ConfusionMatrix(num_classes=self.num_classes)
        self.val_acc_all = []
        self.val_auc_all = []

        self.test_prefix = "test/" if mode == "finetune" else "test/ssl/"
        # self.test_metrics = metrics.clone(prefix=self.test_prefix)
        # self.test_cf = ConfusionMatrix(num_classes=self.num_classes)
        self.test_acc_all = []
        self.test_auc_all = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        # This serves as a point where the model performs the best. It is used later to log values at that epoch.
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:

        ### validation logs ###
        logits = torch.cat(pl_module.all_val_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.val_ds.labels)[:len(logits)]
        scores = self.compute_patch_metrics(logits, labels, prefix=self.val_prefix, scores={})

        # self.val_metrics(logits.softmax(-1), labels) #auc for now
        # self.val_cf(logits.softmax(-1), labels)
        # cf = self.val_cf.compute()
        # tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        # pl_module.log_dict(self.val_metrics.compute(), prog_bar=True, **self.kwargs)
        # pl_module.log(f"{self.val_prefix}{self.mode}_sen", tp / (tp + fn), **self.kwargs)
        # pl_module.log(f"{self.val_prefix}{self.mode}_spe", tn / (tn + fp), **self.kwargs)

        # maximum acc and auc

        # val_auc = self.val_metrics[f'{self.mode}_auc'].compute()
        # val_acc = pl_module.val_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()

        # self.val_auc_all.append(val_auc)
        # self.val_acc_all.append(val_acc)
        #
        # # updating the best epoch
        # # val_best = self.val_acc_all[self.best_epoch] if len(self.val_acc_all)>self.best_epoch else
        # if val_acc >= self.val_acc_all[self.best_epoch]:
        #     self.best_epoch = trainer.current_epoch

        # pl_module.log(f"{self.val_prefix}{self.mode}_auc_best",
        #               self.val_auc_all[self.best_epoch], **self.kwargs)
        # pl_module.log(f"{self.val_prefix}{self.mode}_acc-macro_best",
        #               self.val_acc_all[self.best_epoch], prog_bar=True, **self.kwargs)


        ### test logs ###
        logits = torch.cat(pl_module.all_test_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.test_ds.labels)[:len(logits)]
        scores = self.compute_patch_metrics(logits, labels, prefix=self.test_prefix, scores=scores)

        # self.test_metrics(logits.softmax(-1), labels)
        # self.test_cf(logits.softmax(-1), labels)
        # cf = self.test_cf.compute()
        # tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        # pl_module.log_dict(self.test_metrics.compute(), prog_bar=True, **self.kwargs)
        # pl_module.log(f"{self.test_prefix}{self.mode}_sen", tp / (tp + fn), **self.kwargs)
        # pl_module.log(f"{self.test_prefix}{self.mode}_spe", tn / (tn + fp), **self.kwargs)

        # maximum acc and auc

        self.update_best_epoch(pl_module, trainer, scores=scores)
        scores = self.compute_patch_maxmetrics(prefix=self.val_prefix, scores=scores)
        scores = self.compute_patch_maxmetrics(prefix=self.test_prefix, scores=scores)

        # test_auc = self.test_metrics[f'{self.mode}_auc'].compute()
        # test_acc = pl_module.test_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()
        #
        # self.test_auc_all.append(test_auc)
        # self.test_acc_all.append(test_acc)

        # pl_module.log(f"{self.test_prefix}{self.mode}_auc_best",
        #               self.test_auc_all[self.best_epoch], **self.kwargs)
        # pl_module.log(f"{self.test_prefix}{self.mode}_acc-macro_best",
        #               self.test_acc_all[self.best_epoch], prog_bar=True, **self.kwargs)

        self.log_scores(scores, pl_module)

    def compute_patch_metrics(self, logits, labels, prefix, scores={}):
        scores[f'{prefix}{self.mode}_auc'] = auroc(logits.softmax(-1), labels, num_classes=self.num_classes)

        cf = confusion_matrix(logits.softmax(-1), labels, num_classes=self.num_classes)
        tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        scores[f'{prefix}{self.mode}_sen'] = tp / (tp + fn)
        scores[f'{prefix}{self.mode}_spe'] = tn / (tn + fp)

        return scores

    def update_best_epoch(self, pl_module, trainer, scores):
        val_auc = scores[f'{self.val_prefix}{self.mode}_auc']
        test_auc = scores[f'{self.test_prefix}{self.mode}_auc']
        val_acc = pl_module.val_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()
        test_acc = pl_module.test_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()


        self.val_auc_all.append(val_auc)
        self.val_acc_all.append(val_acc)
        self.test_auc_all.append(test_auc)
        self.test_acc_all.append(test_acc)

        # updating the best epoch
        if val_acc >= self.val_acc_all[self.best_epoch]:
            self.best_epoch = trainer.current_epoch

    def compute_patch_maxmetrics(self, prefix, scores):
        scores[f"{prefix}{self.mode}_auc_best"] = self.val_auc_all[self.best_epoch] if 'val' in prefix else \
            self.test_auc_all[self.best_epoch]

        scores[f"{prefix}{self.mode}_acc-macro_best"] = self.val_acc_all[self.best_epoch] if 'val' in prefix else \
            self.test_acc_all[self.best_epoch]

        return scores

    def log_scores(self, scores, pl_module):
        for key in scores.keys():
            pl_module.log(key, scores[key], on_epoch=True)



    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # self.val_metrics.reset()
        # self.test_metrics.reset()
        # self.val_cf.reset()
        # self.test_cf.reset()

        # reset metrics after sanity checks'
        if trainer.sanity_checking:
            self.val_acc_all = []
            self.val_auc_all = []
            self.test_acc_all = []
            self.test_auc_all = []
