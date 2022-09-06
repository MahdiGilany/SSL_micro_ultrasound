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
        proj_output_dim: int = 512,
        proj_hidden_dim: int = 512,
        sim_loss_weight: float = 25.0,
        var_loss_weight: float = 25.0,
        cov_loss_weight: float = 25.0,
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

        self.inferred_no_centers = 1

        self.val_vicregLoss_all_centers = []
        self.val_simLoss_all_centers = []
        self.val_varLoss_all_centers = []
        self.val_covLoss_all_centers = []

        self.test_vicregLoss_all_centers = []
        self.test_simLoss_all_centers = []
        self.test_varLoss_all_centers = []
        self.test_covLoss_all_centers = []

    @property
    def learnable_params(self) -> List[dict]:
        """Adds projector parameters to the parent's learnable parameters.

        Returns:
            List[dict]: list of learnable parameters.
        """

        extra_learnable_params = [{"params": self.projector.parameters()}]
        return super().learnable_params + extra_learnable_params

    def forward(self, X: torch.Tensor, proj=True) -> Dict[str, Any]:
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
        if proj:
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

        # divide loss by sum of loss weights to compensate -
        # otherwise higher loss weights mean higher effective learning rate.
        vicreg_loss = vicreg_loss / (
            self.var_loss_weight + self.sim_loss_weight + self.cov_loss_weight
        )

        self.log(
            "train/ssl/vicreg_loss",
            vicreg_loss,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ssl/sim_loss",
            all_loss[0],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ssl/var_loss",
            all_loss[1],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )
        self.log(
            "train/ssl/cov_loss",
            all_loss[2],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return vicreg_loss

    def validation_step(
        self, batch: Sequence[Any], batch_idx: int, dataloader_idx: int
    ) -> torch.Tensor:
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
        vicreg_loss, all_losses = vicreg_loss_func(
            z1,
            z2,
            sim_loss_weight=self.sim_loss_weight,
            var_loss_weight=self.var_loss_weight,
            cov_loss_weight=self.cov_loss_weight,
        )
        
        vicreg_loss = vicreg_loss / (
            self.var_loss_weight + self.sim_loss_weight + self.cov_loss_weight
        )
        
        self.logging_combined_centers_losses(dataloader_idx, vicreg_loss, all_losses)

        return vicreg_loss, [], [], [] # these two lists are a workaround to use online_evaluator + metric logger

    def validation_epoch_end(self, outs: List[Any]):
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}

        self.log("val/ssl/vicreg_loss", torch.mean(torch.tensor(self.val_vicregLoss_all_centers)), **kwargs)
        self.log("val/ssl/sim_loss", torch.mean(torch.tensor(self.val_simLoss_all_centers)), **kwargs)
        self.log("val/ssl/var_loss", torch.mean(torch.tensor(self.val_varLoss_all_centers)), **kwargs)
        self.log("val/ssl/cov_loss", torch.mean(torch.tensor(self.val_covLoss_all_centers)), **kwargs)

        self.log("test/ssl/vicreg_loss", torch.mean(torch.tensor(self.test_vicregLoss_all_centers)), **kwargs)
        self.log("test/ssl/sim_loss", torch.mean(torch.tensor(self.test_simLoss_all_centers)), **kwargs)
        self.log("test/ssl/var_loss", torch.mean(torch.tensor(self.test_varLoss_all_centers)), **kwargs)
        self.log("test/ssl/cov_loss", torch.mean(torch.tensor(self.test_covLoss_all_centers)), **kwargs)

    def logging_combined_centers_losses(self, dataloader_idx, vicreg_loss, all_losses):
        self.inferred_no_centers = dataloader_idx + 1 \
            if dataloader_idx + 1 > self.inferred_no_centers \
            else self.inferred_no_centers

        if dataloader_idx < self.inferred_no_centers/2.:
            # all these losses are macro to the centers
            self.val_vicregLoss_all_centers.append(vicreg_loss)
            self.val_simLoss_all_centers.append(all_losses[0])
            self.val_varLoss_all_centers.append(all_losses[1])
            self.val_covLoss_all_centers.append(all_losses[2])

        else:
            # all these losses are macro to the centers
            self.test_vicregLoss_all_centers.append(vicreg_loss)
            self.test_simLoss_all_centers.append(all_losses[0])
            self.test_varLoss_all_centers.append(all_losses[1])
            self.test_covLoss_all_centers.append(all_losses[2])