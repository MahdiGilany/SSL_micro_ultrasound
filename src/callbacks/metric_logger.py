from typing import Any, Dict, Optional, Sequence, Tuple, Union, Literal
from pytorch_lightning.utilities.types import STEP_OUTPUT

import numpy as np
import wandb

import torch
from pytorch_lightning import Callback, LightningModule, Trainer
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC, CatMetric, Recall, Specificity
from torchmetrics.functional import accuracy, confusion_matrix, auroc


class MetricLogger(Callback):
    """Computes and logs all patch-wise metrics
        the model that uses this callback requires:

        # todo: all information that is assumed to be available can be obtained in on_validation_batch_end...
        Requirements:
            # - considers that pl_module.all_val_online_logits variable is available and contarins all logits
            # - considers that pl_module.all_test_online_logits variable is available and contarins all logits
            # - considers that pl_module.val_metrics['finetune(or)online_acc_macro'] variable is available
            # - considers that pl_module.test_metrics['finetune(or)online_acc_macro'] variable is available
            # - trainer.datamodule.val_ds.labels
            # - trainer.datamodule.test_ds.labels
            - trainer.datamodule.val_ds.core_lengths
            - trainer.datamodule.test_ds.core_lengths
        Assumption:
            - val and test centers are all the centers in datamodule.cohort_specifier
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
        # self.val_acc_all = []
        # self.val_auc_all = []
        # self.val_core_auc_all = []
        # self.val_core_acc_all = []

        self.test_prefix = "test/" if mode == "finetune" else "test/ssl/"
        # self.test_acc_all = []
        # self.test_auc_all = []
        # self.test_core_auc_all = []
        # self.test_core_acc_all = []

        self.setup_flag = True
        # self.all_val_test_mlp_logits = {}

        # self.all_centers_val_logits = []
        # self.all_centers_val_labels = []
        # self.val_macroLoss_all_centers = []

        # self.all_centers_test_logits = []
        # self.all_centers_test_labels = []
        # self.test_macroLoss_all_centers = []


        # # metrics for logging
        # self.metrics = MetricCollection({
        #     mode + '_acc_macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
        #     mode + '_auc': AUROC(num_classes=self.num_classes),
        # })
        # self.all_val_test_logits = {}
        #
        # self.all_centers_val_logits = []
        # self.all_centers_val_labels = []
        # self.val_macroLoss_all_centers = []
        #
        # self.all_centers_test_logits = []
        # self.all_centers_test_labels = []
        # self.test_macroLoss_all_centers = []

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

            trainer.datamodule.cohort_specifier = trainer.datamodule.cohort_specifier \
                if isinstance(trainer.datamodule.cohort_specifier, list) \
                else [trainer.datamodule.cohort_specifier]

            self.instantiate_metrics(trainer, pl_module)
            self.initialize_best_metrics_dict(trainer, pl_module)
            self.initialize_patch_logits_dict(trainer, pl_module)
            self.get_coreLengths_coreLabels(trainer, pl_module)

    def get_metrics(self, mode):
        metrics = MetricCollection({
            mode + '_acc_macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
            mode + '_auc': AUROC(num_classes=self.num_classes),
            mode + '_sen': Recall(multiclass=False),
            mode + '_spe': Specificity(multiclass=False),
        })
        return metrics

    def instantiate_metrics(self, trainer, pl_module):
        """
        The following defines torch metric objects for each center separately in val and test and for combination
        of centers as well
        """
        pl_module.val_patch_metrics_centers_dict = {}
        pl_module.val_core_metrics_centers_dict = {}
        pl_module.test_patch_metrics_centers_dict = {}
        pl_module.test_core_metrics_centers_dict = {}

        # pl_module.val_patch_microMetrics_all_centers = None
        # pl_module.val_core_microMetrics_all_centers = None
        # pl_module.test_patch_microMetrics_all_centers = None
        # pl_module.test_core_microMetrics_all_centers = None

        for i, center in enumerate(trainer.datamodule.cohort_specifier):
            # val
            prefix = self.val_prefix + center + '/'
            pl_module.val_patch_metrics_centers_dict[center] = \
                self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
            pl_module.val_core_metrics_centers_dict[center] = \
                self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
            # test
            prefix = self.test_prefix + center + '/'
            pl_module.test_patch_metrics_centers_dict[center] = \
                self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
            pl_module.test_core_metrics_centers_dict[center] = \
                self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)

        # if isinstance(trainer.datamodule.val_ds, dict) and isinstance(trainer.datamodule.test_ds, dict):
        #     # defining metrics for each center individually
        #     # val
        #     for center in trainer.datamodule.val_ds.keys():
        #         prefix = self.val_prefix + center + '/'
        #         pl_module.val_patch_metrics_centers_dict[center] = \
        #             self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        #         pl_module.val_core_metrics_centers_dict[center] = \
        #             self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
        #     # test
        #     for center in trainer.datamodule.test_ds.keys():
        #         prefix = self.test_prefix + center + '/'
        #         pl_module.test_patch_metrics_centers_dict[center] = \
        #             self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        #         pl_module.test_core_metrics_centers_dict[center] = \
        #             self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
        #
        #
        # elif isinstance(trainer.datamodule.val_ds, dict) or isinstance(trainer.datamodule.test_ds, dict):
        #     raise ValueError("both val_ds and test_ds should use the same centers")
        #
        #
        # else:
        #     center = trainer.datamodule.cohort_specifier
        #
        #     # val
        #     prefix = self.val_prefix + center + '/'
        #     pl_module.val_patch_metrics_centers_dict[center] = \
        #         self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        #     pl_module.val_core_metrics_centers_dict[center] = \
        #         self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
        #     # test
        #     prefix = self.test_prefix + center + '/'
        #     pl_module.test_patch_metrics_centers_dict[center] = \
        #         self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        #     pl_module.test_core_metrics_centers_dict[center] = \
        #         self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)

        center = 'all' # all is mico metric meaning combination of centers and reporting micro performance
        # defining metric for combination of centers
        # for example: val/ssl/finetune_macro_acc is patch macro accuracy of all centers combined microly:)

        # val
        prefix = self.val_prefix
        pl_module.val_patch_metrics_centers_dict['all'] = \
            self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        pl_module.val_core_metrics_centers_dict['all'] = \
            self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
        # test
        prefix = self.test_prefix
        pl_module.test_patch_metrics_centers_dict['all'] = \
            self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        pl_module.test_core_metrics_centers_dict['all'] = \
            self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)

        # pl_module.val_patch_microMetrics_all_centers = \
        #     self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        # pl_module.val_core_microMetrics_all_centers = \
        #     self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)
        # test
        # prefix = self.test_prefix
        # pl_module.test_patch_microMetrics_all_centers = \
        #     self.get_metrics(self.mode).clone(prefix=prefix).to(pl_module.device)
        # pl_module.test_core_microMetrics_all_centers = \
        #     self.get_metrics(self.mode + '_core').clone(prefix=prefix).to(pl_module.device)

    def initialize_best_metrics_dict(self, trainer, pl_module):
        self.val_allEpochs_patchAuc_metrics_dict = {}
        self.val_allEpochs_coreAuc_metrics_dict = {}
        self.test_allEpochs_patchAuc_metrics_dict = {}
        self.test_allEpochs_coreAuc_metrics_dict = {}

        self.val_allEpochs_patchAcc_metrics_dict = {}
        self.val_allEpochs_coreAcc_metrics_dict = {}
        self.test_allEpochs_patchAcc_metrics_dict = {}
        self.test_allEpochs_coreAcc_metrics_dict = {}

        for i, center in enumerate(trainer.datamodule.cohort_specifier + ['all']): # all should be after all centers
            self.val_allEpochs_patchAuc_metrics_dict[center] = []
            self.val_allEpochs_coreAuc_metrics_dict[center] = []
            self.test_allEpochs_patchAuc_metrics_dict[center] = []
            self.test_allEpochs_coreAuc_metrics_dict[center] = []

            self.val_allEpochs_patchAcc_metrics_dict[center] = []
            self.val_allEpochs_coreAcc_metrics_dict[center] = []
            self.test_allEpochs_patchAcc_metrics_dict[center] = []
            self.test_allEpochs_coreAcc_metrics_dict[center] = []

    def initialize_patch_logits_dict(self, trainer, pl_module):
        """
        This function defines dictionaries for memorizing logits
        that can be used for finding corewise metrics
        """
        self.val_patch_logits_center_dict = {}
        self.val_patch_labels_center_dict = {}
        self.test_patch_logits_center_dict = {}
        self.test_patch_labels_center_dict = {}

        # self.val_patch_logits_all_centers = []
        # self.val_patch_labels_all_centers = []
        # self.test_patch_logits_all_centers = []
        # self.test_patch_labels_all_centers = []

        for i, center in enumerate(trainer.datamodule.cohort_specifier + ['all']): # all should be after all centers
            self.val_patch_logits_center_dict[center] = []
            self.val_patch_labels_center_dict[center] = []
            # test
            self.test_patch_logits_center_dict[center] = []
            self.test_patch_labels_center_dict[center] = []

        # if isinstance(trainer.datamodule.val_ds, dict) and isinstance(trainer.datamodule.test_ds, dict):
        #     # defining metrics for each center individually
        #     # val
        #     for center in trainer.datamodule.val_ds.keys():
        #         self.val_patch_logits_center_dict[center] = []
        #         self.val_patch_labels_center_dict[center] = []
        #     # test
        #     for center in trainer.datamodule.test_ds.keys():
        #         self.test_patch_logits_center_dict[center] = []
        #         self.test_patch_labels_center_dict[center] = []
        #
        #
        # elif isinstance(trainer.datamodule.val_ds, dict) or isinstance(trainer.datamodule.test_ds, dict):
        #     raise ValueError("both val_ds and test_ds should use the same centers")
        #
        #
        # else:
        #     center = trainer.datamodule.cohort_specifier
        #
        #     # val
        #     self.val_patch_logits_center_dict[center] = []
        #     self.val_patch_labels_center_dict[center] = []
        #     # test
        #     self.test_patch_logits_center_dict[center] = []
        #     self.test_patch_labels_center_dict[center] = []

    def get_coreLengths_coreLabels(self, trainer, pl_module):
        """
        This function memorizes core lengths and core labels for each center separately and for all centers together
        """
        self.val_core_lengths_center_dict = {'all': []}
        self.val_core_labels_center_dict = {'all': []}

        self.test_core_lengths_center_dict = {'all': []}
        self.test_core_labels_center_dict = {'all': []}

        for i, center in enumerate(trainer.datamodule.cohort_specifier):
            # getting corelens and labels from datamodule.val_ds
            if isinstance(trainer.datamodule.val_ds, dict):
                val_corelen_cur_center = torch.tensor(trainer.datamodule.val_ds[center].core_lengths)
                test_corelen_cur_center = torch.tensor(trainer.datamodule.test_ds[center].core_lengths)
                val_corelabel_cur_center = torch.tensor(trainer.datamodule.val_ds[center].core_labels)
                test_corelabel_cur_center = torch.tensor(trainer.datamodule.test_ds[center].core_labels)
            else:
                val_corelen_cur_center = torch.tensor(trainer.datamodule.val_ds.core_lengths)
                test_corelen_cur_center = torch.tensor(trainer.datamodule.test_ds.core_lengths)
                val_corelabel_cur_center = torch.tensor(trainer.datamodule.val_ds.core_labels)
                test_corelabel_cur_center = torch.tensor(trainer.datamodule.test_ds.core_labels)

            self.val_core_lengths_center_dict[center] = val_corelen_cur_center
            self.test_core_lengths_center_dict[center] = test_corelen_cur_center
            self.val_core_labels_center_dict[center] = val_corelabel_cur_center
            self.test_core_labels_center_dict[center] = test_corelabel_cur_center

            self.val_core_lengths_center_dict['all'].append(val_corelen_cur_center)
            self.test_core_lengths_center_dict['all'].append(test_corelen_cur_center)
            self.val_core_labels_center_dict['all'].append(val_corelabel_cur_center)
            self.test_core_labels_center_dict['all'].append(test_corelabel_cur_center)

        self.val_core_lengths_center_dict['all'] = torch.cat(self.val_core_lengths_center_dict['all'])
        self.test_core_lengths_center_dict['all'] = torch.cat(self.test_core_lengths_center_dict['all'])
        self.val_core_labels_center_dict['all'] = torch.cat(self.val_core_labels_center_dict['all'])
        self.test_core_labels_center_dict['all'] = torch.cat(self.test_core_labels_center_dict['all'])

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

        if dataloader_idx < len(pl_module.val_patch_metrics_centers_dict) - 1:
            # val
            val_patch_metrics_cur_center = list(pl_module.val_patch_metrics_centers_dict.values())[dataloader_idx]

            # computing patch metrics
            # logging pytorch lightning MetricCollection using log dict for only ONE center
            logits_cur_center = out1.detach().cpu()
            labels_cur_center = out2.detach().cpu()
            val_patch_metrics_cur_center(logits_cur_center.softmax(-1), labels_cur_center)
            # pl_module.log_dict(val_patch_metrics_cur_center, **kwargs)

            # memorizing logits and labels for each center separately and for all centers together
            list_logits_cur_center = list(self.val_patch_logits_center_dict.values())[dataloader_idx]
            list_labels_cur_center = list(self.val_patch_labels_center_dict.values())[dataloader_idx]
            list_logits_cur_center.append(logits_cur_center)
            list_labels_cur_center.append(labels_cur_center)
            self.val_patch_logits_center_dict['all'].append(logits_cur_center)
            self.val_patch_labels_center_dict['all'].append(labels_cur_center)

            # # calculating metrics of all centers combined
            # if dataloader_idx == len(pl_module.val_patch_metrics_centers_dict) - 2:
            #     logits = torch.cat(self.val_patch_logits_center_dict['all'])
            #     labels = torch.cat(self.val_patch_labels_center_dict['all'])
            #     pl_module.val_patch_metrics_centers_dict['all'](logits.softmax(-1), labels)
            #     # pl_module.log_dict(pl_module.val_patch_metrics_centers_dict['all'], **kwargs)
        else:
            # test
            idx = dataloader_idx - len(pl_module.val_patch_metrics_centers_dict) + 1
            test_patch_metrics_cur_center = list(pl_module.test_patch_metrics_centers_dict.values())[idx]

            # computing patch metrics
            # logging pytorch lightning MetricCollection using log dict for only ONE center
            logits_cur_center = out1.detach().cpu()
            labels_cur_center = out2.detach().cpu()
            test_patch_metrics_cur_center(logits_cur_center.softmax(-1), labels_cur_center)
            # pl_module.log_dict(test_patch_metrics_cur_center, **kwargs)

            # memorizing logits and labels for each center separately and for all centers together
            list_logits_cur_center = list(self.test_patch_logits_center_dict.values())[idx]
            list_labels_cur_center = list(self.test_patch_labels_center_dict.values())[idx]
            list_logits_cur_center.append(logits_cur_center)
            list_labels_cur_center.append(labels_cur_center)
            self.test_patch_logits_center_dict['all'].append(logits_cur_center)
            self.test_patch_labels_center_dict['all'].append(labels_cur_center)

            # # calculating metrics of all centers combined
            # if dataloader_idx == \
            #         len(pl_module.val_patch_metrics_centers_dict) + len(pl_module.test_patch_metrics_centers_dict) - 3:
            #     logits = torch.cat(self.test_patch_logits_center_dict['all'])
            #     labels = torch.cat(self.test_patch_labels_center_dict['all'])
            #     pl_module.test_patch_metrics_centers_dict['all'](logits.softmax(-1), labels)
            #     # pl_module.log_dict(pl_module.test_patch_metrics_centers_dict['all'], **kwargs)
        # for i, val_patch_metrics in enumerate(pl_module.val_patch_metrics_centers_dict.values()):
        #     logits = outputs[1]
        #     labels = outputs[2]
        #
        #     logits_cur_center = torch.cat(logits[i])
        #     labels_cur_center = torch.cat(labels[i])
        #
        #     # computing patch metrics
        #     val_patch_metrics(logits_cur_center.softmax(-1), labels_cur_center)
        #     # logging pytorch lightning MetricCollection using log dict for only ONE center
        #     pl_module.log_dict(val_patch_metrics, on_epoch=True)
        #
        # for i, test_patch_metrics in enumerate(pl_module.test_patch_metrics_centers_dict.values()):
        #     logits = outputs[1]
        #     labels = outputs[2]
        #
        #     logits_cur_center = torch.cat(logits[i])
        #     labels_cur_center = torch.cat(labels[i])
        #
        #     # computing patch metrics
        #     test_patch_metrics(logits_cur_center.softmax(-1), labels_cur_center)
        #     # logging pytorch lightning MetricCollection using log dict for only ONE center
        #     pl_module.log_dict(test_patch_metrics, on_epoch=True)

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        """
        Patch wise metrics are already computed and logged. Core wise is remaining + finding maximum metrics.
        """

        # ### patchwise ###
        self.log_patch_metrics(trainer, pl_module)
        # logits = torch.cat(pl_module.all_val_online_logits).detach().cpu()
        # labels = torch.tensor(trainer.datamodule.val_ds.labels)[:len(logits)]
        # scores = self.compute_patch_metrics(logits, labels, prefix=self.val_prefix, scores={})
        #
        # logits = torch.cat(pl_module.all_test_online_logits).detach().cpu()
        # labels = torch.tensor(trainer.datamodule.test_ds.labels)[:len(logits)]
        # scores = self.compute_patch_metrics(logits, labels, prefix=self.test_prefix, scores=scores)

        ### corewise ###
        self.find_core_logits()
        self.compute_log_core_metrics(trainer, pl_module)
        # scores = self.compute_core_metrics(test_core_probs, test_core_labels, prefix=self.test_prefix, scores=scores)

        ### max metrics ###
        self.update_best_epoch(trainer, pl_module)
        self.log_maxMetrics(trainer, pl_module)

        ### logging scater plots ###
        # todo: add scatter plots
        # self.log_core_scatter(trainer, pl_module)

        # scores = self.compute_patch_maxmetrics(prefix=self.val_prefix, scores=scores)
        # scores = self.compute_patch_maxmetrics(prefix=self.test_prefix, scores=scores)
        #
        # scores = self.compute_core_maxmetrics(prefix=self.val_prefix, scores=scores)
        # scores = self.compute_core_maxmetrics(prefix=self.test_prefix, scores=scores)
        #
        # ### loging all ###
        # self.log_scores(pl_module, scores)
        # self.log_core_scatter(trainer, val_core_probs, test_core_probs)
    def log_patch_metrics(self, trainer, pl_module):
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}

        for i, center in enumerate(pl_module.val_patch_metrics_centers_dict.keys()):
            val_patch_metrics_cur_center = list(pl_module.val_patch_metrics_centers_dict.values())[i]
            test_patch_metrics_cur_center = list(pl_module.test_patch_metrics_centers_dict.values())[i]

            if center == 'all':
                val_logits = torch.cat(self.val_patch_logits_center_dict['all'])
                val_labels = torch.cat(self.val_patch_labels_center_dict['all'])
                test_logits = torch.cat(self.test_patch_logits_center_dict['all'])
                test_labels = torch.cat(self.test_patch_labels_center_dict['all'])
                val_patch_metrics_cur_center(val_logits.softmax(-1), val_labels)
                test_patch_metrics_cur_center(test_logits.softmax(-1), test_labels)

            pl_module.log_dict(val_patch_metrics_cur_center.compute(), **kwargs)
            pl_module.log_dict(test_patch_metrics_cur_center.compute(), **kwargs)

    def find_core_logits(self):
        self.val_core_logits_center_dict = {}
        self.test_core_logits_center_dict = {}

        for i, center in enumerate(self.val_patch_logits_center_dict.keys()):
            val_logits_cur_center = torch.cat(self.val_patch_logits_center_dict[center])
            test_logits_cur_center = torch.cat(self.test_patch_logits_center_dict[center])

            val_preds_cur_center = val_logits_cur_center.argmax(dim=1).detach().cpu()
            test_preds_cur_center = test_logits_cur_center.argmax(dim=1).detach().cpu()

            val_corelen_cur_center = self.val_core_lengths_center_dict[center]
            test_corelen_cur_center = self.test_core_lengths_center_dict[center]


            # get core logits
            self.val_core_logits_center_dict[center] = \
                self.aggregate_patch_preds(val_preds_cur_center, val_corelen_cur_center)
            self.test_core_logits_center_dict[center] = \
                self.aggregate_patch_preds(test_preds_cur_center, test_corelen_cur_center)

        # # computing corewise metrics and logging.
        # corelen_val = trainer.datamodule.val_ds.core_lengths
        # corelen_test = trainer.datamodule.test_ds.core_lengths
        #
        # # all val and test preds in order
        # all_val_logits = torch.cat(pl_module.all_val_online_logits)
        # all_test_logits = torch.cat(pl_module.all_test_online_logits)
        # all_val_preds = all_val_logits.argmax(dim=1).detach().cpu()
        # all_test_preds = all_test_logits.argmax(dim=1).detach().cpu()
        #
        # # find a label for each core in val and test
        # all_val_core_probs = self.get_core_probs(all_val_preds, corelen_val)
        # all_test_core_probs = self.get_core_probs(all_test_preds, corelen_test)

        # # all core labels
        # all_val_coretargets = torch.tensor(trainer.datamodule.val_ds.core_labels)[: len(all_val_core_probs)]
        # all_test_coretargets = torch.tensor(trainer.datamodule.test_ds.core_labels)[: len(all_test_core_probs)]

        # return all_val_core_probs, all_val_coretargets, all_test_core_probs, all_test_coretargets

    def aggregate_patch_preds(self, all_val_preds, corelen_val):
        """This function takes the mean of all patch preds inside a core as prediction of that core."""
        all_corepreds = []
        corelen_cumsum = torch.cumsum(torch.tensor([0] + list(corelen_val)), dim=0)

        for i, value in enumerate(corelen_cumsum):
            if i == 0: # or val > len(all_val_preds):
                continue

            minus1_idx = corelen_cumsum[i - 1]
            core_preds = all_val_preds[minus1_idx:value]
            all_corepreds.append(core_preds.sum() / len(core_preds))

        return torch.tensor(all_corepreds)

    def compute_log_core_metrics(self, trainer, pl_module):
        """
        computing and logging all core metrics
        """
        for i, center in enumerate(pl_module.val_core_metrics_centers_dict.keys()):
            val_cur_core_metric = list(pl_module.val_core_metrics_centers_dict.values())[i]
            val_core_cur_logits = self.val_core_logits_center_dict[center]
            val_core_cur_labels = self.val_core_labels_center_dict[center]

            val_notnan = ~torch.isnan(val_core_cur_logits)
            val_core_cur_logits = val_core_cur_logits[val_notnan]
            val_core_cur_logits = torch.stack([1.- val_core_cur_logits, val_core_cur_logits], dim=1)
            val_cur_core_metric(val_core_cur_logits.softmax(-1), val_core_cur_labels[val_notnan])
            pl_module.log_dict(val_cur_core_metric.compute(), **self.kwargs)

            test_cur_core_metric = list(pl_module.test_core_metrics_centers_dict.values())[i]
            test_core_cur_logits = self.test_core_logits_center_dict[center]
            test_core_cur_labels = self.test_core_labels_center_dict[center]

            test_notnan = ~torch.isnan(test_core_cur_logits)
            test_core_cur_logits = test_core_cur_logits[test_notnan]
            test_core_cur_logits = torch.stack([1. - test_core_cur_logits, test_core_cur_logits], dim=1)
            test_cur_core_metric(test_core_cur_logits.softmax(-1), test_core_cur_labels[test_notnan])
            pl_module.log_dict(test_cur_core_metric.compute(), **self.kwargs)

        # scores[f'{prefix}{self.mode}_core_auc'] = auroc(probs, labels)
        # scores[f'{prefix}{self.mode}_core_acc_macro'] = accuracy(probs, labels, average='macro',
        #                                                          num_classes=self.num_classes, multiclass=True)
        #
        # cf = confusion_matrix(probs, labels, num_classes=self.num_classes)
        # tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]
        #
        # scores[f'{prefix}{self.mode}_core_sen'] = tp / (tp + fn)
        # scores[f'{prefix}{self.mode}_core_spe'] = tn / (tn + fp)
        #
        # return scores

    def update_best_epoch(self, trainer, pl_module):

        for i, center in enumerate(self.val_allEpochs_patchAuc_metrics_dict.keys()):
            val_patch_auc_cur_center = pl_module.val_patch_metrics_centers_dict[center][f'{self.mode}_auc'].compute().detach().cpu()
            val_core_auc_cur_center = pl_module.val_core_metrics_centers_dict[center][f'{self.mode}_core_auc'].compute().detach().cpu()
            test_patch_auc_cur_center = pl_module.test_patch_metrics_centers_dict[center][f'{self.mode}_auc'].compute().detach().cpu()
            test_core_auc_cur_center = pl_module.test_core_metrics_centers_dict[center][f'{self.mode}_core_auc'].compute().detach().cpu()

            val_patch_acc_cur_center = pl_module.val_patch_metrics_centers_dict[center][f'{self.mode}_acc_macro'].compute().detach().cpu()
            val_core_acc_cur_center = pl_module.val_core_metrics_centers_dict[center][f'{self.mode}_core_acc_macro'].compute().detach().cpu()
            test_patch_acc_cur_center = pl_module.test_patch_metrics_centers_dict[center][f'{self.mode}_acc_macro'].compute().detach().cpu()
            test_core_acc_cur_center = pl_module.test_core_metrics_centers_dict[center][f'{self.mode}_core_acc_macro'].compute().detach().cpu()


            self.val_allEpochs_patchAuc_metrics_dict[center].append(val_patch_auc_cur_center)
            self.val_allEpochs_coreAuc_metrics_dict[center].append(val_core_auc_cur_center)
            self.test_allEpochs_patchAuc_metrics_dict[center].append(test_patch_auc_cur_center)
            self.test_allEpochs_coreAuc_metrics_dict[center].append(test_core_auc_cur_center)

            self.val_allEpochs_patchAcc_metrics_dict[center].append(val_patch_acc_cur_center)
            self.val_allEpochs_coreAcc_metrics_dict[center].append(val_core_acc_cur_center)
            self.test_allEpochs_patchAcc_metrics_dict[center].append(test_patch_acc_cur_center)
            self.test_allEpochs_coreAcc_metrics_dict[center].append(test_core_acc_cur_center)

        # # updating the best epoch
        val_patch_auc_sf = self.val_allEpochs_patchAuc_metrics_dict['all']
        if val_patch_auc_sf[-1] >= val_patch_auc_sf[self.best_epoch]:
            self.best_epoch = trainer.current_epoch

        # # patchwise
        # val_auc = scores[f'{self.val_prefix}{self.mode}_auc']
        # test_auc = scores[f'{self.test_prefix}{self.mode}_auc']
        # val_acc = pl_module.val_metrics[f'{self.mode}_acc_macro'].compute().detach().cpu()
        # test_acc = pl_module.test_metrics[f'{self.mode}_acc_macro'].compute().detach().cpu()
        #
        # # corewise
        # val_core_auc = scores[f'{self.val_prefix}{self.mode}_core_auc']
        # test_core_auc = scores[f'{self.test_prefix}{self.mode}_core_auc']
        # val_core_acc = scores[f'{self.val_prefix}{self.mode}_core_acc_macro']
        # test_core_acc = scores[f'{self.test_prefix}{self.mode}_core_acc_macro']
        #
        #
        # self.val_auc_all.append(val_auc)
        # self.val_acc_all.append(val_acc)
        # self.test_auc_all.append(test_auc)
        # self.test_acc_all.append(test_acc)
        #
        # self.val_core_auc_all.append(val_core_auc)
        # self.val_core_acc_all.append(val_core_acc)
        # self.test_core_auc_all.append(test_core_auc)
        # self.test_core_acc_all.append(test_core_acc)
        #
        # # updating the best epoch
        # if val_acc >= self.val_acc_all[self.best_epoch]:
        #     self.best_epoch = trainer.current_epoch

    def log_maxMetrics(self, trainer, pl_module):
        max_scores = {}

        for i, center in enumerate(trainer.datamodule.cohort_specifier):
            max_scores[f"{self.val_prefix}{center}/{self.mode}_auc_best"] = self.val_allEpochs_patchAuc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.val_prefix}{center}/{self.mode}_core_auc_best"] = self.val_allEpochs_coreAuc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.test_prefix}{center}/{self.mode}_auc_best"] = self.test_allEpochs_patchAuc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.test_prefix}{center}/{self.mode}_core_auc_best"] = self.test_allEpochs_coreAuc_metrics_dict[center][self.best_epoch]

            max_scores[f"{self.val_prefix}{center}/{self.mode}_acc_best"] = self.val_allEpochs_patchAcc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.val_prefix}{center}/{self.mode}_core_acc_best"] = self.val_allEpochs_coreAcc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.test_prefix}{center}/{self.mode}_acc_best"] = self.test_allEpochs_patchAcc_metrics_dict[center][self.best_epoch]
            max_scores[f"{self.test_prefix}{center}/{self.mode}_core_acc_best"] = self.test_allEpochs_coreAcc_metrics_dict[center][self.best_epoch]


        max_scores[f"{self.val_prefix}{self.mode}_auc_best"] = self.val_allEpochs_patchAuc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.val_prefix}{self.mode}_core_auc_best"] = self.val_allEpochs_coreAuc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.test_prefix}{self.mode}_auc_best"] = self.test_allEpochs_patchAuc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.test_prefix}{self.mode}_core_auc_best"] = self.test_allEpochs_coreAuc_metrics_dict['all'][self.best_epoch]

        max_scores[f"{self.val_prefix}{self.mode}_acc_best"] = self.val_allEpochs_patchAcc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.val_prefix}{self.mode}_core_acc_best"] = self.val_allEpochs_coreAcc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.test_prefix}{self.mode}_acc_best"] = self.test_allEpochs_patchAcc_metrics_dict['all'][self.best_epoch]
        max_scores[f"{self.test_prefix}{self.mode}_core_acc_best"] = self.test_allEpochs_coreAcc_metrics_dict['all'][self.best_epoch]

        pl_module.log_dict(max_scores, **self.kwargs)

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
        for i, center in enumerate(self.val_patch_logits_center_dict.keys()):
            pl_module.val_patch_metrics_centers_dict[center].reset()
            pl_module.val_core_metrics_centers_dict[center].reset()
            pl_module.test_patch_metrics_centers_dict[center].reset()
            pl_module.test_core_metrics_centers_dict[center].reset()
            self.val_patch_logits_center_dict[center] = []
            self.val_patch_labels_center_dict[center] = []
            self.test_patch_logits_center_dict[center] = []
            self.test_patch_labels_center_dict[center] = []
            self.val_core_logits_center_dict = {}
            self.test_core_logits_center_dict = {}

        if trainer.sanity_checking:
            self.initialize_best_metrics_dict(trainer, pl_module)

        # # val
        # for i, center in enumerate(pl_module.val_patch_metrics_centers_dict):
        #     pl_module.val_patch_metrics_centers_dict[center].reset()
        #     pl_module.val_core_metrics_centers_dict[center].reset()
        #     self.val_patch_logits_center_dict[center] = []
        #     self.val_patch_labels_center_dict[center] = []
        # # test
        # for i, center in enumerate(pl_module.test_patch_metrics_centers_dict):
        #     pl_module.test_patch_metrics_centers_dict[center].reset()
        #     pl_module.test_core_metrics_centers_dict[center].reset()
        #     self.test_patch_logits_center_dict[center] = []
        #     self.test_patch_labels_center_dict[center] = []

        # pl_module.val_patch_microMetrics_all_centers.reset()
        # pl_module.val_core_microMetrics_all_centers.reset()
        # pl_module.test_patch_microMetrics_all_centers.reset()
        # pl_module.test_core_microMetrics_all_centers.reset()
        #
        # self.val_patch_logits_all_centers = []
        # self.val_patch_labels_all_centers = []
        # self.test_patch_logits_all_centers = []
        # self.test_patch_labels_all_centers = []

        # if trainer.sanity_checking:
        #     self.val_acc_all = []
        #     self.val_auc_all = []
        #     self.test_acc_all = []
        #     self.test_auc_all = []
        #
        #     self.val_core_auc_all = []
        #     self.val_core_acc_all = []
        #     self.test_core_auc_all = []
        #     self.test_core_acc_all = []

    # def compute_patch_metrics(self, logits, labels, prefix, scores={}):
    #     scores[f'{prefix}{self.mode}_auc'] = auroc(logits.softmax(-1), labels, num_classes=self.num_classes)
    #
    #     cf = confusion_matrix(logits.softmax(-1), labels, num_classes=self.num_classes)
    #     tp, fp, tn, fn = cf[1, 1], cf[0, 1], cf[0, 0], cf[1, 0]
    #
    #     scores[f'{prefix}{self.mode}_sen'] = tp / (tp + fn)
    #     scores[f'{prefix}{self.mode}_spe'] = tn / (tn + fp)
    #
    #     return scores



    # def compute_patch_maxmetrics(self, prefix, scores):
    #     scores[f"{prefix}{self.mode}_auc_best"] = self.val_auc_all[self.best_epoch] if 'val' in prefix else \
    #         self.test_auc_all[self.best_epoch]
    #
    #     scores[f"{prefix}{self.mode}_acc_macro_best"] = self.val_acc_all[self.best_epoch] if 'val' in prefix else \
    #         self.test_acc_all[self.best_epoch]
    #
    #     return scores

    # def compute_core_maxmetrics(self, prefix, scores):
    #     scores[f"{prefix}{self.mode}_core_auc_best"] = self.val_core_auc_all[self.best_epoch] if 'val' in prefix \
    #         else self.test_core_auc_all[self.best_epoch]
    #
    #     scores[f"{prefix}{self.mode}_core_acc_macro_best"] = self.val_core_acc_all[self.best_epoch] if 'val' in prefix \
    #         else self.test_core_acc_all[self.best_epoch]
    #
    #     return scores

    # def log_scores(self, pl_module, scores):
    #     for key in scores.keys():
    #         pl_module.log(key, scores[key], on_epoch=True)

