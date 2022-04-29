from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import wandb

from src.models.components.backbones import *

from ..exact_ssl_module import ExactSSLModule
from ..losses.vicreg_loss import vicreg_loss_func


class VICReg(ExactSSLModule):
    def __init__(
        self,
        proj_output_dim: int,
        proj_hidden_dim: int,
        sim_loss_weight: float = 25.,
        var_loss_weight: float = 25.,
        cov_loss_weight: float = 25.,
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

        # # for applications that need feature vector only
        # if not self.training:
        #     return out

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
        z1, z2 = out["z"]

        # ------- vicreg loss -------
        vicreg_loss, all_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        self.log("train/ssl/vicreg_loss", vicreg_loss, on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ssl/sim_loss", all_loss[0], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ssl/var_loss", all_loss[1], on_step=False, on_epoch=True, sync_dist=True)
        self.log("train/ssl/cov_loss", all_loss[2], on_step=False, on_epoch=True, sync_dist=True)

        return vicreg_loss

    def validation_step(self, batch: Sequence[Any], batch_idx: int, dataloader_idx: int) -> torch.Tensor:
        """Validation step for VICReg reusing BaseMethod validation step.

        Args:
            batch (Sequence[Any]): a batch of data in the format of [img_indexes, [X], Y], where
                [X] is a list of size num_crops containing batches of images.
            batch_idx (int): index of the batch.

        Returns:
            torch.Tensor: total loss composed of VICReg loss and classification loss.
        """

        out = super().validation_step(batch, batch_idx, dataloader_idx)
        z1, z2 = out["z"]

        # ------- vicreg loss -------
        vicreg_loss, all_loss = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )

        if dataloader_idx == 0:
            self.log("val/ssl/vicreg_loss", vicreg_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/ssl/sim_loss", all_loss[0], on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/ssl/var_loss", all_loss[1], on_step=False, on_epoch=True, sync_dist=True)
            self.log("val/ssl/cov_loss", all_loss[2], on_step=False, on_epoch=True, sync_dist=True)
        elif dataloader_idx == 1:
            self.log("test/ssl/vicreg_loss", vicreg_loss, on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/ssl/sim_loss", all_loss[0], on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/ssl/var_loss", all_loss[1], on_step=False, on_epoch=True, sync_dist=True)
            self.log("test/ssl/cov_loss", all_loss[2], on_step=False, on_epoch=True, sync_dist=True)

        return vicreg_loss
