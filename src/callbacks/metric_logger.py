from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal

import numpy as np
import wandb

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

        self.val_prefix = "val/" if mode == "finetune" else "val/ssl/"
        self.val_acc_all = []
        self.val_auc_all = []
        self.val_core_auc_all = []
        self.val_core_acc_all = []

        self.test_prefix = "test/" if mode == "finetune" else "test/ssl/"
        self.test_acc_all = []
        self.test_auc_all = []
        self.test_core_auc_all = []
        self.test_core_acc_all = []

    def setup(self, trainer: Trainer, pl_module: LightningModule, stage: Optional[str] = None) -> None:
        # This serves as a point where the model performs the best. It is used later to log values at that epoch.
        self.best_epoch = 0

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        ### patchwise ###
        logits = torch.cat(pl_module.all_val_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.val_ds.labels)[:len(logits)]
        scores = self.compute_patch_metrics(logits, labels, prefix=self.val_prefix, scores={})

        logits = torch.cat(pl_module.all_test_online_logits).detach().cpu()
        labels = torch.tensor(trainer.datamodule.test_ds.labels)[:len(logits)]
        scores = self.compute_patch_metrics(logits, labels, prefix=self.test_prefix, scores=scores)

        ### corewise ###
        val_core_probs, var_core_labels, test_core_probs, test_core_labels = self.get_core_logits_labels(trainer,
                                                                                                         pl_module)
        scores = self.compute_core_metrics(val_core_probs, var_core_labels, prefix=self.val_prefix, scores=scores)
        scores = self.compute_core_metrics(test_core_probs, test_core_labels, prefix=self.test_prefix, scores=scores)

        ### max metrics ###
        self.update_best_epoch(pl_module, trainer, scores=scores)

        scores = self.compute_patch_maxmetrics(prefix=self.val_prefix, scores=scores)
        scores = self.compute_patch_maxmetrics(prefix=self.test_prefix, scores=scores)

        scores = self.compute_core_maxmetrics(prefix=self.val_prefix, scores=scores)
        scores = self.compute_core_maxmetrics(prefix=self.test_prefix, scores=scores)

        ### loging all ###
        self.log_scores(pl_module, scores)
        self.log_core_scatter(trainer, val_core_probs, test_core_probs)

    def compute_patch_metrics(self, logits, labels, prefix, scores={}):
        scores[f'{prefix}{self.mode}_auc'] = auroc(logits.softmax(-1), labels, num_classes=self.num_classes)

        cf = confusion_matrix(logits.softmax(-1), labels, num_classes=self.num_classes)
        tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        scores[f'{prefix}{self.mode}_sen'] = tp / (tp + fn)
        scores[f'{prefix}{self.mode}_spe'] = tn / (tn + fp)

        return scores

    def update_best_epoch(self, pl_module, trainer, scores):
        # patchwise
        val_auc = scores[f'{self.val_prefix}{self.mode}_auc']
        test_auc = scores[f'{self.test_prefix}{self.mode}_auc']
        val_acc = pl_module.val_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()
        test_acc = pl_module.test_metrics[f'{self.mode}_acc-macro'].compute().detach().cpu()

        # corewise
        val_core_auc = scores[f'{self.val_prefix}{self.mode}_core_auc']
        test_core_auc = scores[f'{self.test_prefix}{self.mode}_core_auc']
        val_core_acc = scores[f'{self.val_prefix}{self.mode}_core_acc-macro']
        test_core_acc = scores[f'{self.test_prefix}{self.mode}_core_acc-macro']


        self.val_auc_all.append(val_auc)
        self.val_acc_all.append(val_acc)
        self.test_auc_all.append(test_auc)
        self.test_acc_all.append(test_acc)

        self.val_core_auc_all.append(val_core_auc)
        self.val_core_acc_all.append(val_core_acc)
        self.test_core_auc_all.append(test_core_auc)
        self.test_core_acc_all.append(test_core_acc)

        # updating the best epoch
        if val_acc >= self.val_acc_all[self.best_epoch]:
            self.best_epoch = trainer.current_epoch

    def compute_patch_maxmetrics(self, prefix, scores):
        scores[f"{prefix}{self.mode}_auc_best"] = self.val_auc_all[self.best_epoch] if 'val' in prefix else \
            self.test_auc_all[self.best_epoch]

        scores[f"{prefix}{self.mode}_acc-macro_best"] = self.val_acc_all[self.best_epoch] if 'val' in prefix else \
            self.test_acc_all[self.best_epoch]

        return scores

    def get_core_logits_labels(self, trainer, pl_module):
        # computing corewise metrics and logging.
        corelen_val = trainer.datamodule.val_ds.core_lengths
        corelen_test = trainer.datamodule.test_ds.core_lengths

        # all val and test preds in order
        all_val_logits = torch.cat(pl_module.all_val_online_logits)
        all_test_logits = torch.cat(pl_module.all_test_online_logits)
        all_val_preds = all_val_logits.argmax(dim=1).detach().cpu()
        all_test_preds = all_test_logits.argmax(dim=1).detach().cpu()

        # find a label for each core in val and test
        all_val_core_probs = self.get_core_probs(all_val_preds, corelen_val)
        all_test_core_probs = self.get_core_probs(all_test_preds, corelen_test)

        # all core labels
        all_val_coretargets = torch.tensor(trainer.datamodule.val_ds.core_labels)[: len(all_val_core_probs)]
        all_test_coretargets = torch.tensor(trainer.datamodule.test_ds.core_labels)[: len(all_test_core_probs)]

        return all_val_core_probs, all_val_coretargets, all_test_core_probs, all_test_coretargets

    def get_core_probs(self, all_val_preds, corelen_val):
        """This function takes the mean of all patches inside a core as prediction of that core."""
        all_val_corepreds = []
        corelen_cumsum = torch.cumsum(torch.tensor([0] + corelen_val), dim=0)

        for i, val in enumerate(corelen_cumsum):
            if i == 0 or val > len(all_val_preds):
                continue

            val_minus1 = corelen_cumsum[i - 1]
            core_preds = all_val_preds[val_minus1:val]
            all_val_corepreds.append(core_preds.sum() / len(core_preds))

        return torch.tensor(all_val_corepreds)

    def compute_core_metrics(self, probs, labels, prefix, scores={}):
        scores[f'{prefix}{self.mode}_core_auc'] = auroc(probs, labels)
        scores[f'{prefix}{self.mode}_core_acc-macro'] = accuracy(probs, labels, average='macro',
                                                                 num_classes=self.num_classes, multiclass=True)

        cf = confusion_matrix(probs, labels, num_classes=self.num_classes)
        tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]

        scores[f'{prefix}{self.mode}_core_sen'] = tp / (tp + fn)
        scores[f'{prefix}{self.mode}_core_spe'] = tn / (tn + fp)

        return scores

    def compute_core_maxmetrics(self, prefix, scores):
        scores[f"{prefix}{self.mode}_core_auc_best"] = self.val_core_auc_all[self.best_epoch] if 'val' in prefix \
            else self.test_core_auc_all[self.best_epoch]

        scores[f"{prefix}{self.mode}_core_acc-macro_best"] = self.val_core_acc_all[self.best_epoch] if 'val' in prefix \
            else self.test_core_acc_all[self.best_epoch]

        return scores

    def log_scores(self, pl_module, scores):
        for key in scores.keys():
            pl_module.log(key, scores[key], on_epoch=True)

    def log_core_scatter(
            self,
            trainer,
            val_core_probs,
            test_core_probs,
    ):
        val_core_inv = trainer.datamodule.val_ds.core_inv
        data = [[x, y] for (x, y) in zip(val_core_inv, val_core_probs)]
        table = wandb.Table(columns=["True_inv", "Pred_inv"], data=data)
        wandb.log({f"{self.val_prefix}{self.mode}_core_scatter": wandb.plot.scatter(table, "True_inv", "Pred_inv")})

        test_core_inv = trainer.datamodule.test_ds.core_inv
        data = [[x, y] for (x, y) in zip(test_core_inv, test_core_probs)]
        table = wandb.Table(columns=["True_inv", "Pred_inv"], data=data)
        wandb.log({f"{self.test_prefix}{self.mode}_core_scatter": wandb.plot.scatter(table, "True_inv", "Pred_inv")})

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reset metrics after sanity checks'
        if trainer.sanity_checking:
            self.val_acc_all = []
            self.val_auc_all = []
            self.test_acc_all = []
            self.test_auc_all = []

            self.val_core_auc_all = []
            self.val_core_acc_all = []
            self.test_core_auc_all = []
            self.test_core_acc_all = []
