import warnings
from argparse import ArgumentParser
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_optimizer as trch_opt
import wandb
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MaxMetric
from torchmetrics.classification.accuracy import Accuracy
from torchmetrics.classification.confusion_matrix import ConfusionMatrix

from src.models.components.backbones import *
from src.models.components.simple_dense_net import SimpleDenseNet
from src.models.components.utils import LARSWrapper, weighted_mean
from src.models.components.vicreg_loss import vicreg_loss_func


def static_lr(
    get_lr: Callable, param_group_indexes: Sequence[int], lrs_to_replace: Sequence[float]
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class ExactSSLModule(LightningModule):
    """Example of LightningModule for SSL Exact classification.

    A LightningModule organizes your PyTorch code into 5 sections:
        - Computations (init).
        - Train loop (training_step)
        - Validation loop (validation_step)
        - Test loop (test_step)
        - Optimizers (configure_optimizers)

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html
    """

    _SUPPORTED_BACKBONES = {
        "resnet10": resnet10,
        "resnet18": resnet18,
        "resnet50": resnet50,
    }

    def __init__(
        self,
        backbone: str,
        lr: float = 0.0001,
        weight_decay: float = 0.000,
        epoch: int = 100,
        batch_size: int = 32,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        # macro means balanced accuracy
        self.acc_obj = Accuracy() #average='macro', num_classes=2)

        # for logging best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # training related
        self.num_classes = 2
        self.max_epochs = epoch
        self.batch_size = batch_size
        self.optimizer = 'adam'
        self.lr = lr
        self.classifier_lr = 0.5
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = 0 # todo: check larger batch size.
        self.scheduler = "warmup_cosine"

        self.lars = False
        self.exclude_bias_n_norm = True
        self.extra_optimizer_args = {}
        self.lr_decay_steps = [60, 80]
        self.min_lr = 0.0
        self.warmup_start_lr = 0.0
        self.warmup_epochs = 10
        self.scheduler_interval = "step"
        self.num_large_crops = 2
        self.num_small_crops = 0
        self.eta_lars = 0.02
        self.grad_clip_lars = False
        self.lr_decay_steps = False

        self._num_training_steps = None

        # multicrop
        self.num_crops = self.num_large_crops + self.num_small_crops

        # all the other parameters
        self.extra_args = []

        # turn on multicrop if there are small crops
        self.multicrop = self.num_small_crops != 0

        assert backbone in ExactSSLModule._SUPPORTED_BACKBONES
        self.base_model = self._SUPPORTED_BACKBONES[backbone]

        self.backbone_name = backbone

        self.backbone = self.base_model()
        if "resnet" in self.backbone_name:
            self.features_dim = self.backbone.inplanes
            # remove fc layer
            self.backbone.fc = nn.Identity()
        else:
            self.features_dim = self.backbone.num_features

        self.classifier = nn.Linear(self.features_dim, self.num_classes)

        if self.scheduler_interval == "step":
            warnings.warn(
                f"Using scheduler_interval={self.scheduler_interval} might generate "
                "issues when resuming a checkpoint."
            )

    def forward(self, X) -> Dict:
        """Basic forward method. Children methods should call this function, modify the ouputs
        (without deleting anything) and return it.

        Args:
            X (torch.Tensor): batch of images in tensor format.

        Returns:
            Dict: dict of logits and features.
        """

        # if not self.no_channel_last:
        #     X = X.to(memory_format=torch.channels_last)
        feats = self.backbone(X)
        logits = self.classifier(feats.detach())
        return {"logits": logits, "feats": feats}

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """

        out = self(X)
        logits = out["logits"]

        loss = F.cross_entropy(logits, targets, ignore_index=-1)

        acc = self.acc_obj(torch.argmax(logits, dim=1), targets)
        self.acc_obj.reset()

        out.update({"loss": loss, "acc": acc, "logits": logits})
        return out

    def training_step(self, batch: Any, batch_idx: int):

        # lr = self.onecyc_scheduler.get_last_lr()
        #
        # self.log("lr", lr[0], on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/loss", loss, on_step=False, on_epoch=True, prog_bar=False)
        # self.log("train/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # return {"loss": loss, "preds": preds, "targets": targets}

        """Training step for pytorch lightning. It does all the shared operations, such as
        forwarding the crops, computing logits and computing statistics.

        Args:
            batch (List[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size self.num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            Dict[str, Any]: dict with the classification loss, features and logits.
        """

        X, targets = batch

        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self._base_shared_step(x, targets) for x in X[:self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        # loss and stats
        outs["loss"] = sum(outs["loss"]) / self.num_large_crops
        outs["acc"] = sum(outs["acc"]) / self.num_large_crops

        self.log("train/ssl/linear-loss", outs["loss"], on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("train/ssl/linear-acc", outs["acc"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.scheduler is not None:
            lr = self.scheduler_obj.get_last_lr()
            self.log("lr", lr[0], on_step=False, on_epoch=True, prog_bar=False)

        return outs

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        X, targets = batch
        batch_size = targets.size(0)

        outs = self._base_shared_step(X, targets)

        # if self.knn_eval and not self.trainer.sanity_checking:
        #     self.knn(test_features=out.pop("feats").detach(), test_targets=targets.detach())


        # if dataloader_idx == 0:
        # #     log val metrics
        #
        #
        # elif dataloader_idx == 1:
        # #     log test metrics

        return {"loss": outs["loss"], "acc": outs["acc"], "logits": outs["logits"], "batch_size": batch_size}

    def validation_epoch_end(self, outs: List[Dict[str, Any]]):
        """Averages the losses and accuracies of all the validation batches. This is needed because
        the last batch can be smaller than the others, slightly skewing the metrics.

        Args:
            outs (List[Dict[str, Any]]): list of outputs of the validation step.
        """

        val_loss = weighted_mean(outs[0], "loss", "batch_size")
        val_acc = weighted_mean(outs[0], "acc", "batch_size")

        test_loss = weighted_mean(outs[1], "loss", "batch_size")
        test_acc = weighted_mean(outs[1], "acc", "batch_size")

        # saving all preds for corewise callback: val and test
        self.all_val_preds = torch.cat([i['logits'] for i in outs[0]], dim=0).argmax(dim=1).detach().cpu().numpy()
        self.all_test_preds = torch.cat([i['logits'] for i in outs[1]], dim=0).argmax(dim=1).detach().cpu().numpy()

        self.log("val/ssl/linear-loss", val_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("val/ssl/linear-acc", val_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        self.val_acc_best.update(val_acc)
        self.log("val/ssl/linear-acc_best", self.val_acc_best.compute(), on_step=False, on_epoch=True, prog_bar=True,
                 sync_dist=True)

        self.log("test/ssl/linear-loss", test_loss, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.log("test/ssl/linear-acc", test_acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def on_epoch_end(self):
        # reset metrics after sanity checks'
        if self.trainer.sanity_checking:
            self.val_acc_best.reset()

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if self._num_training_steps is None:
            len_ds = len(self.trainer.datamodule.train_ds)
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    @property
    def learnable_params(self) -> List[Dict[str, Any]]:
        """Defines learnable parameters for the base class.

        Returns:
            List[Dict[str, Any]]:
                list of dicts containing learnable parameters and possible settings.
        """

        return [
            {"name": "backbone", "params": self.backbone.parameters()},
            {
                "name": "classifier",
                "params": self.classifier.parameters(),
                "lr": self.classifier_lr,
                "weight_decay": 0,
            },
        ]

    def configure_optimizers(self) -> Tuple[List, List]:
        """Collects learnable parameters and configures the optimizer and learning rate scheduler.

        Returns:
            Tuple[List, List]: two lists containing the optimizer and the scheduler.
        """

        # collect learnable parameters
        idxs_no_scheduler = [
            i for i, m in enumerate(self.learnable_params) if m.pop("static_lr", False)
        ]

        # select optimizer
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD
        elif self.optimizer == "adam":
            optimizer = torch.optim.Adam
        elif self.optimizer == "adamw":
            optimizer = torch.optim.AdamW
        else:
            raise ValueError(f"{self.optimizer} not in (sgd, adam, adamw)")

        # create optimizer
        optimizer = optimizer(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optimizer == "sgd", "LARS is only compatible with SGD."
            optimizer = LARSWrapper(
                optimizer,
                eta=self.eta_lars,
                clip=self.grad_clip_lars,
                exclude_bias_n_norm=self.exclude_bias_n_norm,
            )

        if self.scheduler is None:
            return optimizer

        if self.scheduler == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.lr,
                    eta_min=self.min_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
            self.scheduler_obj = scheduler["scheduler"]
        elif self.scheduler == "step":
            scheduler = MultiStepLR(optimizer, self.lr_decay_steps)
            self.scheduler_obj = scheduler
        else:
            raise ValueError(f"{self.scheduler} not in (warmup_cosine, cosine, step)")

        if idxs_no_scheduler:
            partial_fn = partial(
                static_lr,
                get_lr=scheduler["scheduler"].get_lr
                if isinstance(scheduler, dict)
                else scheduler.get_lr,
                param_group_indexes=idxs_no_scheduler,
                lrs_to_replace=[self.lr] * len(idxs_no_scheduler),
            )
            if isinstance(scheduler, dict):
                scheduler["scheduler"].get_lr = partial_fn
            else:
                scheduler.get_lr = partial_fn

        return [optimizer], [scheduler]


class VICReg(ExactSSLModule):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float,
        var_loss_weight: float,
        cov_loss_weight: float,
        **kwargs
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            sim_loss_weight (float): weight of the invariance term.
            var_loss_weight (float): weight of the variance term.
            cov_loss_weight (float): weight of the covariance term.
        """

        super().__init__(**kwargs)

        self.sim_loss_weight = sim_loss_weight
        self.var_loss_weight = var_loss_weight
        self.cov_loss_weight = cov_loss_weight

        # projector
        self.projector = nn.Sequential(
            nn.Linear(self.features_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_hidden_dim),
            nn.BatchNorm1d(proj_hidden_dim),
            nn.ReLU(),
            nn.Linear(proj_hidden_dim, proj_output_dim),
        )

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor) -> Dict[str, Any]:
        """Performs the forward pass of the backbone and the projector.

        Args:
            X (torch.Tensor): a batch of images in the tensor format.

        Returns:
            Dict[str, Any]: a dict containing the outputs of the parent and the projected features.
        """

        out = super().forward(X)
        z = self.projector(out["feats"])
        out.update({"z": z})
        return out

    def training_step(self, batch: Sequence[Any], batch_idx: int) -> torch.Tensor:
        """Training step for VICReg reusing BaseMethod training step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().training_step(batch, batch_idx)
        class_loss = out["loss"]
        z1, z2 = out["z"]

        # ------- vicreg loss -------
        vicreg_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train/ssl/vicreg_loss", vicreg_loss, on_step=False, on_epoch=True, sync_dist=True)

        return vicreg_loss + class_loss
