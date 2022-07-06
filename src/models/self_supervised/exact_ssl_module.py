import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple, Literal

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics import MinMetric
from torchmetrics.classification.accuracy import Accuracy

from src.models.components.backbones import *
from src.models.components.utils import LARSWrapper, weighted_mean

import torch_optimizer


def static_lr(
    get_lr: Callable,
    param_group_indexes: Sequence[int],
    lrs_to_replace: Sequence[float],
):
    lrs = get_lr()
    for idx, lr in zip(param_group_indexes, lrs_to_replace):
        lrs[idx] = lr
    return lrs


class ExactSSLModule(LightningModule):
    """Example of LightningModule for SSL Exact classification.

    For using this module:
        - datamodule should have object variable: train_ds

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
        "resnet10_feat_dim_256": resnet10_feat_dim_256,
        "resnet10_feat_dim_128": resnet10_feat_dim_128,
        "resnet10_feat_dim_64": resnet10_feat_dim_64,
        "resnet10_compressed_to_ndim": resnet10_compressed_to_ndim,
        "resnet10_tiny_compressed_to_3dim": resnet10_tiny_compressed_to_3dim,
    }

    # Models in this list should have fc as the last layer, which will be removed.
    # models OUTSIDE this list should return features directly, rather than class
    # logits, during forward(X), as there will be no model.fc = Identity() applied
    _RESNET_BASED_BACKBONES = [
        "resnet10",
        "resnet18",
        "resnet50",
        "resnet10_feat_dim_256",
        "resnet10_feat_dim_128",
        "resnet10_feat_dim_64",
    ]

    def __init__(
            self,
            backbone: str,
            lr: float = 0.0001,
            weight_decay: float = 0.000,
            optim_algo: Literal["Adam", "Novograd"] = "Adam",
            epoch: int = 100,
            batch_size: int = 32,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # it also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.all_val_online_logits = []
        self.all_test_online_logits = []

        # for logging best so far validation accuracy
        self.val_loss_best = MinMetric()

        # training related
        self.num_classes = 2
        self.max_epochs = epoch
        self.batch_size = batch_size
        self.optim_algo = optim_algo
        self.lr = lr
        self.weight_decay = weight_decay
        self.accumulate_grad_batches = 0  # todo: check larger batch size.
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
        if self.backbone_name in ExactSSLModule._RESNET_BASED_BACKBONES:
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
        return {"feats": feats}

    def _base_shared_step(self, X: torch.Tensor, targets: torch.Tensor) -> Dict:
        """Forwards a batch of images X and computes the classification loss, the logits, the
        features, acc@1 and acc@5.

        Args:
            X (torch.Tensor): batch of images in tensor format.
            targets (torch.Tensor): batch of labels for X.

        Returns:
            Dict: dict containing the classification loss, logits, features, acc@1 and acc@5.
        """
        # todo: not add any step here. This function should be removed
        out = self(X)
        return out

    def training_step(self, batch: Any, batch_idx: int):
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

        outs = [self._base_shared_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        if self.scheduler is not None:
            lr = self.scheduler_obj.get_last_lr()
            self.log("lr", lr[0], on_step=False, on_epoch=True, prog_bar=False)

        return outs

    def training_epoch_end(self, outputs: List[Any]):
        # `outputs` is a list of dicts returned from `training_step()`
        pass

    def validation_step(self, batch: Any, batch_idx: int, dataloader_idx: int):
        # todo we can have shared step for train and val
        X, targets = batch
        X = [X] if isinstance(X, torch.Tensor) else X

        # check that we received the desired number of crops
        assert len(X) == self.num_crops

        outs = [self._base_shared_step(x, targets) for x in X[: self.num_large_crops]]
        outs = {k: [out[k] for out in outs] for k in outs[0].keys()}

        return outs

    def validation_epoch_end(self, outs: List[Any]):
        pass

    def test_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0):
        pass

    def on_epoch_end(self):
        # reset all saved logits
        self.all_val_online_logits = []
        self.all_test_online_logits = []

        # reset metrics after sanity checks'
        if self.trainer.sanity_checking:
            self.val_loss_best.reset()

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

        # create optimizer
        optim_algo = self.set_optim_algo()
        optimizer = optim_algo(
            self.learnable_params,
            lr=self.lr,
            weight_decay=self.weight_decay,
            **self.extra_optimizer_args,
        )
        # optionally wrap with lars
        if self.lars:
            assert self.optim_algo == "SGD", "LARS is only compatible with SGD."
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
                    warmup_start_lr=self.warmup_start_lr
                    if self.warmup_epochs > 0
                    else self.lr,
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

    def set_optim_algo(self, **kwargs):
        optim_algo = {
            'SGD': torch.optim.SGD,
            'Adam': torch.optim.Adam,
            'AdamW': torch.optim.AdamW,
            'Novograd': torch_optimizer.NovoGrad
        }

        if self.optim_algo not in optim_algo.keys():
            raise ValueError(f"{self.optim_algo} not in {optim_algo.keys()}")

        return optim_algo[self.optim_algo]
