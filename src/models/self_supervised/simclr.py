import pytorch_lightning as pl
from .exact_ssl_module import ExactSSLModule
import torch
from torch.nn import functional as F
import logging
from typing import Any, Callable, Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import wandb
from src.models.components.backbones import *

from .exact_ssl_module import ExactSSLModule


def simclr_loss(z1, z2, temperature=1.0):

    n = z1.shape[0]

    Z = torch.concat([z1, z2], dim=0)

    # normalize
    Z = F.normalize(Z, dim=-1)

    # scores (i j) - the cosine distance between zi and zj
    scores = (Z @ Z.T) / temperature

    logging.debug(f"\nscores = {scores}")

    # get rid of diagonals
    LARGE_NUM = 1e9
    scores[torch.arange(2 * n), torch.arange(2 * n)] = -LARGE_NUM

    logging.debug(f"\nscores = {scores}")

    # formulate simclr loss as cross entropy loss
    # when predicting which of the other reps is the positive match

    targets = (
        torch.concat([torch.arange(n) + n, torch.arange(n)]).long().to(scores.device)
    )

    logging.debug(f"\n{targets=}")
    probs = scores.softmax(-1)

    logging.debug(f"\n{probs=}")

    loss = F.cross_entropy(probs, targets)

    return loss


class SimCLR(ExactSSLModule):
    def __init__(
        self,
        temperature: float = 1.0,
        proj_hidden_dim: int = 128,
        proj_output_dim: int = 64,
        **kwargs,
    ):
        """Implements VICReg (https://arxiv.org/abs/2105.04906)

        Args:
            proj_output_dim (int): number of dimensions of the projected features.
            proj_hidden_dim (int): number of neurons in the hidden layers of the projector.
            temperature: parameter for simclr loss
        """

        super().__init__(**kwargs)

        self.temperature = temperature

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

        self.tracked_loss_names = ["simclr_loss"]

        self.losses_all_centers = {}
        self.losses_all_centers["val"] = {k: [] for k in self.tracked_loss_names}
        self.losses_all_centers["test"] = {k: [] for k in self.tracked_loss_names}

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

        loss = simclr_loss(z1, z2, self.temperature)
        loss_dict = {"simclr_loss": loss}

        [
            self.log(
                f"train/ssl/{loss_name}",
                loss_value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
            for loss_name, loss_value in loss_dict.items()
        ]

        return loss

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

        loss_dict = {"simclr_loss": simclr_loss(z1, z2, self.temperature)}

        self.logging_combined_centers_losses(dataloader_idx, loss_dict)

        return (
            loss_dict["simclr_loss"],
            [],
            [],
            [],
        )  # these two lists are a workaround to use online_evaluator + metric logger

    def validation_epoch_end(self, outs: List[Any]):
        kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }

        [
            self.log(
                f"val/ssl/{loss_name}", torch.mean(torch.tensor(loss_list)), **kwargs
            )
            for loss_name, loss_list in self.losses_all_centers["val"].items()
        ]

        [loss_list.clear() for loss_list in self.losses_all_centers["val"].values()]

        [
            self.log(
                f"test/ssl/{loss_name}", torch.mean(torch.tensor(loss_list)), **kwargs
            )
            for loss_name, loss_list in self.losses_all_centers["test"].items()
        ]

        [loss_list.clear() for loss_list in self.losses_all_centers["test"].values()]

    def logging_combined_centers_losses(self, dataloader_idx, loss_dict):

        assert all([key in self.tracked_loss_names for key in loss_dict])

        self.inferred_no_centers = (
            dataloader_idx + 1
            if dataloader_idx + 1 > self.inferred_no_centers
            else self.inferred_no_centers
        )

        val_or_test = (
            "val" if dataloader_idx < self.inferred_no_centers / 2.0 else "test"
        )

        [
            self.losses_all_centers[val_or_test][loss_name].append(loss_value)
            for loss_name, loss_value in loss_dict.items()
        ]
