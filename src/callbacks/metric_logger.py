from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal
from pytorch_lightning.utilities.types import STEP_OUTPUT

import numpy as np
import wandb

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC, CatMetric, Recall, Specificity
from torchmetrics.functional import accuracy, confusion_matrix, auroc

from src.callbacks.components.metrics import PatchMetricManager, CoreMetricManager


class MetricLogger(Callback):
    """Computes and logs all patch-wise metrics
        the model that uses this callback requires:

        # todo: all information that is assumed to be available can be obtained in on_validation_batch_end...
        Requirements:
            - trainer.datamodule.val_ds.core_labels
            - trainer.datamodule.test_ds.core_labels
            - trainer.datamodule.val_ds.core_lengths
            - trainer.datamodule.test_ds.core_lengths
        Assumption:
            - val and test centers are all the centers in datamodule.cohort_specifier
    """

    def __init__(
            self,
            mode: Literal["online", "finetune"] = "finetune",
            num_classes: int = 2,
            corewise_metrics: bool = True,
            corewise_inv_threshold: float = 0.5,
    ):

        super().__init__()

        self.mode = mode
        self.num_classes = num_classes
        self.corewise_metrics = corewise_metrics
        self.inv_threshold = corewise_inv_threshold
        self.setup_flag = True

    def setup(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        stage: Optional[str] = None,
    ) -> None:
        # This serves as a point where the model performs the best. It is used later to log values at that epoch.
        self.best_epoch = 0
        if self.setup_flag:
            self.setup_flag = False

            self.cohort_specifier = trainer.datamodule.cohort_specifier \
                if isinstance(trainer.datamodule.cohort_specifier, list) \
                else [trainer.datamodule.cohort_specifier]

            self.patch_metric_manager = PatchMetricManager(
                self.cohort_specifier,
                self.mode,
                self.num_classes
            )

            if self.corewise_metrics:
                self.core_metric_manager = CoreMetricManager(
                    trainer.datamodule.val_ds,
                    trainer.datamodule.test_ds,
                    self.cohort_specifier,
                    self.mode,
                    self.num_classes
                )


    def on_validation_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs: Optional[STEP_OUTPUT],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        """
        computing and logging all patch metrics
        """

        out1 = outputs[1] if not isinstance(outputs[1], list) else outputs[1][0]
        out2 = outputs[2] if not isinstance(outputs[2], list) else outputs[2][0]
        out3 = outputs[3] if not isinstance(outputs[3], list) else outputs[3][0]

        logits_cur_center = out1.detach().cpu()
        labels_cur_center = out2.detach().cpu()
        gs_cur_center = torch.stack((out3["primary_grade"], out3["secondary_grade"])).detach().cpu()

        if dataloader_idx < len(self.cohort_specifier):
            # val
            self.patch_metric_manager.update(
                "val",
                self.cohort_specifier[dataloader_idx],
                logits_cur_center.softmax(-1),
                labels_cur_center,
                gs_cur_center
            )
        else:
            # test
            self.patch_metric_manager.update(
                "test",
                self.cohort_specifier[int(dataloader_idx/2.)-1],
                logits_cur_center.softmax(-1),
                labels_cur_center,
                gs_cur_center
            )

    def find_best_epoch(self,):
        metrics_over_epochs = self.patch_metric_manager._getMetric("val", center='ALL', gs='ALL').get_allMetricValues
        auc_key = [key for key in metrics_over_epochs[0].keys() if 'auc' in key][0]
        auc_over_epochs = [metric[auc_key] for metric in metrics_over_epochs]
        return np.argmax(auc_over_epochs)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Patch wise metrics are already computed and logged. Core wise is remaining + finding maximum metrics.
        """

        ### patchwise ###
        self.patch_metric_manager.log(pl_module)

        ### corewise ###
        if self.corewise_metrics:
            val_core_logits, val_core_labels, val_gs = \
                self.core_metric_manager.get_logits_from_patchManager("val", self.patch_metric_manager)
            test_core_logits, test_core_labels, test_gs = \
                self.core_metric_manager.get_logits_from_patchManager("test", self.patch_metric_manager)

            self.core_metric_manager.update("val", val_core_logits, val_core_labels, val_gs)
            self.core_metric_manager.update("test", test_core_logits, test_core_labels, test_gs)
            self.core_metric_manager.log(pl_module)

        ### max metrics ###
        if not trainer.sanity_checking:
            best_epoch = self.find_best_epoch()
            self.patch_metric_manager.log_optimum(pl_module, best_epoch)
            self.core_metric_manager.log_optimum(pl_module, best_epoch)

    def log_core_scatter(self, trainer, pl_module):
        val_core_probs = []
        test_core_probs = []

        val_core_inv = np.asarray(trainer.datamodule.val_ds.core_inv) / 100.0
        data = [[x, y] for (x, y) in zip(val_core_inv, val_core_probs)]
        table = wandb.Table(columns=["True_inv", "Pred_inv"], data=data)
        wandb.log({f"{self.val_prefix}{self.mode}_core_scatter": wandb.plot.scatter(table, "True_inv", "Pred_inv")})

        test_core_inv = np.asarray(trainer.datamodule.test_ds.core_inv) / 100.0
        data = [[x, y] for (x, y) in zip(test_core_inv, test_core_probs)]
        table = wandb.Table(columns=["True_inv", "Pred_inv"], data=data)
        wandb.log({f"{self.test_prefix}{self.mode}_core_scatter": wandb.plot.scatter(table, "True_inv", "Pred_inv")})

    def on_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # reset metrics
        reset_tracker = True if trainer.sanity_checking else False
        self.patch_metric_manager.reset(reset_tracker)

        if self.corewise_metrics:
            self.core_metric_manager.reset(reset_tracker)