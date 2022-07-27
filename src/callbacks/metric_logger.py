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
        # self.kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True}

        # self.val_prefix = "val/" if mode == "finetune" else "val/ssl/"
        # self.test_prefix = "test/" if mode == "finetune" else "test/ssl/"

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

            # self.instantiate_metrics(trainer, pl_module)
            # self.initialize_best_metrics_dict(trainer, pl_module)
            # self.initialize_patch_logits_dict(trainer, pl_module)
            # self.get_coreLengths_coreLabels(trainer, pl_module)

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

    # def get_metrics(self, mode):
    #     metrics = MetricCollection({
    #         mode + '_acc_macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
    #         mode + '_auc': AUROC(num_classes=self.num_classes),
    #         mode + '_sen': Recall(multiclass=False),
    #         mode + '_spe': Specificity(multiclass=False),
    #     })
    #     return metrics

    # def instantiate_metrics(self, trainer, pl_module):
    #     """
    #     The following defines torch metric objects for each center separately in val and test and for combination
    #     of centers as well
    #     """
    #     pl_module.val_patch_metrics_centers_dict = {}
    #     pl_module.val_core_metrics_centers_dict = {}
    #     pl_module.test_patch_metrics_centers_dict = {}
    #     pl_module.test_core_metrics_centers_dict = {}
    #
    #     for i, center in enumerate(trainer.datamodule.cohort_specifier):
    #         # val
    #         prefix = self.val_prefix + center + '/'
    #         pl_module.val_patch_metrics_centers_dict[center] = \
    #             self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
    #         pl_module.val_core_metrics_centers_dict[center] = \
    #             self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
    #         # test
    #         prefix = self.test_prefix + center + '/'
    #         pl_module.test_patch_metrics_centers_dict[center] = \
    #             self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
    #         pl_module.test_core_metrics_centers_dict[center] = \
    #             self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
    #
    #     prefix = self.val_prefix
    #     pl_module.val_patch_metrics_centers_dict['all'] = \
    #         self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
    #     pl_module.val_core_metrics_centers_dict['all'] = \
    #         self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
    #     # test
    #     prefix = self.test_prefix
    #     pl_module.test_patch_metrics_centers_dict['all'] = \
    #         self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
    #     pl_module.test_core_metrics_centers_dict['all'] = \
    #         self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)

    # def initialize_best_metrics_dict(self, trainer, pl_module):
    #     self.val_allEpochs_patchAuc_metrics_dict = {}
    #     self.val_allEpochs_coreAuc_metrics_dict = {}
    #     self.test_allEpochs_patchAuc_metrics_dict = {}
    #     self.test_allEpochs_coreAuc_metrics_dict = {}
    #
    #     self.val_allEpochs_patchAcc_metrics_dict = {}
    #     self.val_allEpochs_coreAcc_metrics_dict = {}
    #     self.test_allEpochs_patchAcc_metrics_dict = {}
    #     self.test_allEpochs_coreAcc_metrics_dict = {}

        # for i, center in enumerate(trainer.datamodule.cohort_specifier + ['all']): # all should be after all centers
        #     self.val_allEpochs_patchAuc_metrics_dict[center] = []
        #     self.val_allEpochs_coreAuc_metrics_dict[center] = []
        #     self.test_allEpochs_patchAuc_metrics_dict[center] = []
        #     self.test_allEpochs_coreAuc_metrics_dict[center] = []
        #
        #     self.val_allEpochs_patchAcc_metrics_dict[center] = []
        #     self.val_allEpochs_coreAcc_metrics_dict[center] = []
        #     self.test_allEpochs_patchAcc_metrics_dict[center] = []
        #     self.test_allEpochs_coreAcc_metrics_dict[center] = []

    # def initialize_patch_logits_dict(self, trainer, pl_module):
    #     """
    #     This function defines dictionaries for memorizing logits
    #     that can be used for finding corewise metrics
    #     """
    #     self.val_patch_logits_center_dict = {}
    #     self.val_patch_labels_center_dict = {}
    #     self.test_patch_logits_center_dict = {}
    #     self.test_patch_labels_center_dict = {}
    #
    #     for i, center in enumerate(trainer.datamodule.cohort_specifier + ['all']): # all should be after all centers
    #         self.val_patch_logits_center_dict[center] = []
    #         self.val_patch_labels_center_dict[center] = []
    #         # test
    #         self.test_patch_logits_center_dict[center] = []
    #         self.test_patch_labels_center_dict[center] = []

    # def get_coreLengths_coreLabels(self, trainer, pl_module):
    #     """
    #     This function memorizes core lengths and core labels for each center separately and for all centers together
    #     """
    #     self.val_core_lengths_center_dict = {'all': []}
    #     self.val_core_labels_center_dict = {'all': []}
    #
    #     self.test_core_lengths_center_dict = {'all': []}
    #     self.test_core_labels_center_dict = {'all': []}
    #
    #     for i, center in enumerate(trainer.datamodule.cohort_specifier):
    #         # getting corelens and labels from datamodule.val_ds
    #         if isinstance(trainer.datamodule.val_ds, dict):
    #             val_corelen_cur_center = torch.tensor(trainer.datamodule.val_ds[center].core_lengths)
    #             test_corelen_cur_center = torch.tensor(trainer.datamodule.test_ds[center].core_lengths)
    #             val_corelabel_cur_center = torch.tensor(trainer.datamodule.val_ds[center].core_labels)
    #             test_corelabel_cur_center = torch.tensor(trainer.datamodule.test_ds[center].core_labels)
    #         else:
    #             val_corelen_cur_center = torch.tensor(trainer.datamodule.val_ds.core_lengths)
    #             test_corelen_cur_center = torch.tensor(trainer.datamodule.test_ds.core_lengths)
    #             val_corelabel_cur_center = torch.tensor(trainer.datamodule.val_ds.core_labels)
    #             test_corelabel_cur_center = torch.tensor(trainer.datamodule.test_ds.core_labels)
    #
    #         self.val_core_lengths_center_dict[center] = val_corelen_cur_center
    #         self.test_core_lengths_center_dict[center] = test_corelen_cur_center
    #         self.val_core_labels_center_dict[center] = val_corelabel_cur_center
    #         self.test_core_labels_center_dict[center] = test_corelabel_cur_center
    #
    #         self.val_core_lengths_center_dict['all'].append(val_corelen_cur_center)
    #         self.test_core_lengths_center_dict['all'].append(test_corelen_cur_center)
    #         self.val_core_labels_center_dict['all'].append(val_corelabel_cur_center)
    #         self.test_core_labels_center_dict['all'].append(test_corelabel_cur_center)
    #
    #     self.val_core_lengths_center_dict['all'] = torch.cat(self.val_core_lengths_center_dict['all'])
    #     self.test_core_lengths_center_dict['all'] = torch.cat(self.test_core_lengths_center_dict['all'])
    #     self.val_core_labels_center_dict['all'] = torch.cat(self.val_core_labels_center_dict['all'])
    #     self.test_core_labels_center_dict['all'] = torch.cat(self.test_core_labels_center_dict['all'])

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
            # val_patch_metrics_cur_center = list(pl_module.val_patch_metrics_centers_dict.values())[dataloader_idx]
            #
            # logits_cur_center = out1.detach().cpu()
            # labels_cur_center = out2.detach().cpu()
            # val_patch_metrics_cur_center(logits_cur_center.softmax(-1), labels_cur_center)
            #
            # list_logits_cur_center = list(self.val_patch_logits_center_dict.values())[dataloader_idx]
            # list_labels_cur_center = list(self.val_patch_labels_center_dict.values())[dataloader_idx]
            # list_logits_cur_center.append(logits_cur_center)
            # list_labels_cur_center.append(labels_cur_center)
            # self.val_patch_logits_center_dict['all'].append(logits_cur_center)
            # self.val_patch_labels_center_dict['all'].append(labels_cur_center)
        else:
            # test
            self.patch_metric_manager.update(
                "test",
                self.cohort_specifier[int(dataloader_idx/2.)-1],
                logits_cur_center.softmax(-1),
                labels_cur_center,
                gs_cur_center
            )
            # idx = dataloader_idx - len(pl_module.val_patch_metrics_centers_dict) + 1
            # test_patch_metrics_cur_center = list(pl_module.test_patch_metrics_centers_dict.values())[idx]
            #
            # logits_cur_center = out1.detach().cpu()
            # labels_cur_center = out2.detach().cpu()
            # test_patch_metrics_cur_center(logits_cur_center.softmax(-1), labels_cur_center)
            #
            # list_logits_cur_center = list(self.test_patch_logits_center_dict.values())[idx]
            # list_labels_cur_center = list(self.test_patch_labels_center_dict.values())[idx]
            # list_logits_cur_center.append(logits_cur_center)
            # list_labels_cur_center.append(labels_cur_center)
            # self.test_patch_logits_center_dict['all'].append(logits_cur_center)
            # self.test_patch_labels_center_dict['all'].append(labels_cur_center)

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

            # self.log_patch_metrics(trainer, pl_module)

        ### corewise ###
        # self.find_core_logits()
        # self.compute_log_core_metrics(trainer, pl_module)


        # self.update_best_epoch(trainer, pl_module)
        # self.log_maxMetrics(trainer, pl_module)

        ### logging scater plots ###
        # todo: add scatter plots
        # self.log_core_scatter(trainer, pl_module)

    # def log_patch_metrics(self, trainer, pl_module):
    #     kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}
    #
    #     for i, center in enumerate(pl_module.val_patch_metrics_centers_dict.keys()):
    #         val_patch_metrics_cur_center = list(pl_module.val_patch_metrics_centers_dict.values())[i]
    #         test_patch_metrics_cur_center = list(pl_module.test_patch_metrics_centers_dict.values())[i]
    #
    #         if center == 'all':
    #             val_logits = torch.cat(self.val_patch_logits_center_dict['all'])
    #             val_labels = torch.cat(self.val_patch_labels_center_dict['all'])
    #             test_logits = torch.cat(self.test_patch_logits_center_dict['all'])
    #             test_labels = torch.cat(self.test_patch_labels_center_dict['all'])
    #             val_patch_metrics_cur_center(val_logits.softmax(-1), val_labels)
    #             test_patch_metrics_cur_center(test_logits.softmax(-1), test_labels)
    #
    #         pl_module.log_dict(val_patch_metrics_cur_center.compute(), **kwargs)
    #         pl_module.log_dict(test_patch_metrics_cur_center.compute(), **kwargs)

    # def find_core_logits(self):
    #     self.val_core_logits_center_dict = {}
    #     self.test_core_logits_center_dict = {}
    #
    #     for i, center in enumerate(self.val_patch_logits_center_dict.keys()):
    #         val_logits_cur_center = torch.cat(self.val_patch_logits_center_dict[center])
    #         test_logits_cur_center = torch.cat(self.test_patch_logits_center_dict[center])
    #
    #         val_preds_cur_center = val_logits_cur_center.argmax(dim=1).detach().cpu()
    #         test_preds_cur_center = test_logits_cur_center.argmax(dim=1).detach().cpu()
    #
    #         val_corelen_cur_center = self.val_core_lengths_center_dict[center]
    #         test_corelen_cur_center = self.test_core_lengths_center_dict[center]
    #
    #
    #         # get core logits
    #         self.val_core_logits_center_dict[center] = \
    #             self.aggregate_patch_preds(val_preds_cur_center, val_corelen_cur_center)
    #         self.test_core_logits_center_dict[center] = \
    #             self.aggregate_patch_preds(test_preds_cur_center, test_corelen_cur_center)

    # def aggregate_patch_preds(self, all_val_preds, corelen_val):
    #     """This function takes the mean of all patch preds inside a core as prediction of that core."""
    #     all_corepreds = []
    #     corelen_cumsum = torch.cumsum(torch.tensor([0] + list(corelen_val)), dim=0)
    #
    #     for i, value in enumerate(corelen_cumsum):
    #         if i == 0: # or val > len(all_val_preds):
    #             continue
    #
    #         minus1_idx = corelen_cumsum[i - 1]
    #         core_preds = all_val_preds[minus1_idx:value]
    #         all_corepreds.append(core_preds.sum() / len(core_preds))
    #
    #     return torch.tensor(all_corepreds)
    #
    # def compute_log_core_metrics(self, trainer, pl_module):
    #     """
    #     computing and logging all core metrics
    #     """
    #     for i, center in enumerate(pl_module.val_core_metrics_centers_dict.keys()):
    #         val_cur_core_metric = list(pl_module.val_core_metrics_centers_dict.values())[i]
    #         val_core_cur_logits = self.val_core_logits_center_dict[center]
    #         val_core_cur_labels = self.val_core_labels_center_dict[center]
    #
    #         val_notnan = ~torch.isnan(val_core_cur_logits)
    #         val_core_cur_logits = val_core_cur_logits[val_notnan]
    #         val_core_cur_logits = torch.stack([1.- val_core_cur_logits, val_core_cur_logits], dim=1)
    #         val_cur_core_metric(val_core_cur_logits.softmax(-1), val_core_cur_labels[val_notnan])
    #         pl_module.log_dict(val_cur_core_metric.compute(), **self.kwargs)
    #
    #         test_cur_core_metric = list(pl_module.test_core_metrics_centers_dict.values())[i]
    #         test_core_cur_logits = self.test_core_logits_center_dict[center]
    #         test_core_cur_labels = self.test_core_labels_center_dict[center]
    #
    #         test_notnan = ~torch.isnan(test_core_cur_logits)
    #         test_core_cur_logits = test_core_cur_logits[test_notnan]
    #         test_core_cur_logits = torch.stack([1. - test_core_cur_logits, test_core_cur_logits], dim=1)
    #         test_cur_core_metric(test_core_cur_logits.softmax(-1), test_core_cur_labels[test_notnan])
    #         pl_module.log_dict(test_cur_core_metric.compute(), **self.kwargs)

    # def update_best_epoch(self, trainer, pl_module):
    #
    #     for i, center in enumerate(self.val_allEpochs_patchAuc_metrics_dict.keys()):
    #         val_patch_auc_cur_center = pl_module.val_patch_metrics_centers_dict[center][f'{self.mode}_auc'].compute().detach().cpu()
    #         val_core_auc_cur_center = pl_module.val_core_metrics_centers_dict[center][f'{self.mode}_core_auc'].compute().detach().cpu()
    #         test_patch_auc_cur_center = pl_module.test_patch_metrics_centers_dict[center][f'{self.mode}_auc'].compute().detach().cpu()
    #         test_core_auc_cur_center = pl_module.test_core_metrics_centers_dict[center][f'{self.mode}_core_auc'].compute().detach().cpu()
    #
    #         val_patch_acc_cur_center = pl_module.val_patch_metrics_centers_dict[center][f'{self.mode}_acc_macro'].compute().detach().cpu()
    #         val_core_acc_cur_center = pl_module.val_core_metrics_centers_dict[center][f'{self.mode}_core_acc_macro'].compute().detach().cpu()
    #         test_patch_acc_cur_center = pl_module.test_patch_metrics_centers_dict[center][f'{self.mode}_acc_macro'].compute().detach().cpu()
    #         test_core_acc_cur_center = pl_module.test_core_metrics_centers_dict[center][f'{self.mode}_core_acc_macro'].compute().detach().cpu()
    #
    #
    #         self.val_allEpochs_patchAuc_metrics_dict[center].append(val_patch_auc_cur_center)
    #         self.val_allEpochs_coreAuc_metrics_dict[center].append(val_core_auc_cur_center)
    #         self.test_allEpochs_patchAuc_metrics_dict[center].append(test_patch_auc_cur_center)
    #         self.test_allEpochs_coreAuc_metrics_dict[center].append(test_core_auc_cur_center)
    #
    #         self.val_allEpochs_patchAcc_metrics_dict[center].append(val_patch_acc_cur_center)
    #         self.val_allEpochs_coreAcc_metrics_dict[center].append(val_core_acc_cur_center)
    #         self.test_allEpochs_patchAcc_metrics_dict[center].append(test_patch_acc_cur_center)
    #         self.test_allEpochs_coreAcc_metrics_dict[center].append(test_core_acc_cur_center)
    #
    #     # # updating the best epoch
    #     val_patch_auc_sf = self.val_allEpochs_patchAuc_metrics_dict['all']
    #     if val_patch_auc_sf[-1] >= val_patch_auc_sf[self.best_epoch]:
    #         self.best_epoch = trainer.current_epoch

    # def log_maxMetrics(self, trainer, pl_module):
    #     max_scores = {}
    #
    #     for i, center in enumerate(trainer.datamodule.cohort_specifier):
    #         max_scores[f"{self.val_prefix}{center}/{self.mode}_auc_best"] = self.val_allEpochs_patchAuc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.val_prefix}{center}/{self.mode}_core_auc_best"] = self.val_allEpochs_coreAuc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.test_prefix}{center}/{self.mode}_auc_best"] = self.test_allEpochs_patchAuc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.test_prefix}{center}/{self.mode}_core_auc_best"] = self.test_allEpochs_coreAuc_metrics_dict[center][self.best_epoch]
    #
    #         max_scores[f"{self.val_prefix}{center}/{self.mode}_acc_best"] = self.val_allEpochs_patchAcc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.val_prefix}{center}/{self.mode}_core_acc_best"] = self.val_allEpochs_coreAcc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.test_prefix}{center}/{self.mode}_acc_best"] = self.test_allEpochs_patchAcc_metrics_dict[center][self.best_epoch]
    #         max_scores[f"{self.test_prefix}{center}/{self.mode}_core_acc_best"] = self.test_allEpochs_coreAcc_metrics_dict[center][self.best_epoch]
    #
    #
    #     max_scores[f"{self.val_prefix}{self.mode}_auc_best"] = self.val_allEpochs_patchAuc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.val_prefix}{self.mode}_core_auc_best"] = self.val_allEpochs_coreAuc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.test_prefix}{self.mode}_auc_best"] = self.test_allEpochs_patchAuc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.test_prefix}{self.mode}_core_auc_best"] = self.test_allEpochs_coreAuc_metrics_dict['all'][self.best_epoch]
    #
    #     max_scores[f"{self.val_prefix}{self.mode}_acc_best"] = self.val_allEpochs_patchAcc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.val_prefix}{self.mode}_core_acc_best"] = self.val_allEpochs_coreAcc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.test_prefix}{self.mode}_acc_best"] = self.test_allEpochs_patchAcc_metrics_dict['all'][self.best_epoch]
    #     max_scores[f"{self.test_prefix}{self.mode}_core_acc_best"] = self.test_allEpochs_coreAcc_metrics_dict['all'][self.best_epoch]
    #
    #     pl_module.log_dict(max_scores, **self.kwargs)

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

        # for i, center in enumerate(self.val_patch_logits_center_dict.keys()):
        #     pl_module.val_patch_metrics_centers_dict[center].reset()
        #     pl_module.val_core_metrics_centers_dict[center].reset()
        #     pl_module.test_patch_metrics_centers_dict[center].reset()
        #     pl_module.test_core_metrics_centers_dict[center].reset()
        #     self.val_patch_logits_center_dict[center] = []
        #     self.val_patch_labels_center_dict[center] = []
        #     self.test_patch_logits_center_dict[center] = []
        #     self.test_patch_labels_center_dict[center] = []
        #     self.val_core_logits_center_dict = {}
        #     self.test_core_logits_center_dict = {}

        # if trainer.sanity_checking:
        #     self.initialize_best_metrics_dict(trainer, pl_module)
