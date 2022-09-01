from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal
import pytorch_lightning as pl
from torchmetrics.classification.accuracy import Accuracy
import torch
import torch_optimizer
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR


@dataclass
class SharedStepOutput:
    logits: torch.Tensor
    y: torch.Tensor
    loss: torch.Tensor
    metadata: list


class EvaluationBase(pl.LightningModule):
    """
    Makes the model compatible with logging the validation metrics,
    and allows supervised training by only defining shared_step() method and
    trainable_parameters() method
    """

    def __init__(
        self,
        batch_size: int,
        epochs: int = 100,
        learning_rate: float = 0.1,
        weight_decay: float = 1e-6,
        nesterov: bool = False,
        scheduler_type: str = "cosine",
        decay_epochs: tuple = (60, 80),
        gamma: float = 0.1,
        final_lr: float = 0.0,
        optim_algo: Literal["Adam", "Novograd"] = "Adam",
    ):

        super().__init__()

        self.batch_size = batch_size
        self.train_acc = Accuracy()

        self.learning_rate = learning_rate
        self.nesterov = nesterov
        self.weight_decay = weight_decay

        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.epochs = epochs
        self.final_lr = final_lr
        self.optim_algo = optim_algo

        self.scheduler_interval = "step"
        self.warmup_epochs = 10
        self.warmup_start_lr = 0.0
        self.max_epochs = epochs

        self.inferred_no_centers = 1

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

    def shared_step(self, batch) -> SharedStepOutput:
        raise NotImplementedError

    def get_learnable_parameters(self):
        raise NotImplementedError

    def training_step(self, batch, batch_idx):

        output = self.shared_step(batch)
        self.train_acc(output.logits.softmax(-1), output.y)

        self.log(
            "train/finetune_loss",
            output.loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log(
            "train/finetune_acc",
            self.train_acc.compute(),
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return output.loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        out = self.shared_step(batch)

        self.logging_combined_centers_loss(dataloader_idx, out.loss)

        return out.loss, out.logits, out.y, *out.metadata

    def validation_epoch_end(self, outs):
        kwargs = {
            "on_step": False,
            "on_epoch": True,
            "sync_dist": True,
            "add_dataloader_idx": False,
        }
        self.log(
            "val/finetune_loss",
            torch.mean(torch.tensor(self.val_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )
        self.log(
            "test/finetune_loss",
            torch.mean(torch.tensor(self.test_macroLoss_all_centers)),
            prog_bar=True,
            **kwargs,
        )

    def test_step(self, batch, batch_idx):
        pass

    def on_epoch_end(self):
        self.train_acc.reset()

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

    @property
    def num_training_steps(self) -> int:
        """Compute the number of training steps for each epoch."""

        if not hasattr(self, "_num_training_steps"):
            len_ds = len(self.trainer.datamodule.train_ds)
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    def logging_combined_centers_loss(self, dataloader_idx, loss):
        """macro loss for centers"""
        self.inferred_no_centers = (
            dataloader_idx + 1
            if dataloader_idx + 1 > self.inferred_no_centers
            else self.inferred_no_centers
        )

        if dataloader_idx < self.inferred_no_centers / 2.0:
            self.val_macroLoss_all_centers.append(loss)
        else:
            self.test_macroLoss_all_centers.append(loss)

    def configure_optimizers(self):
        opt_params = self.get_learnable_parameters()
        optim_algo = self.set_optim_algo()
        optimizer = optim_algo(
            opt_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer, self.decay_epochs, gamma=self.gamma
            )
        elif self.scheduler_type == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, self.epochs, eta_min=self.final_lr  # total epochs to run
            )
        elif self.scheduler_type == "warmup_cosine":
            scheduler = {
                "scheduler": LinearWarmupCosineAnnealingLR(
                    optimizer,
                    warmup_epochs=self.warmup_epochs * self.num_training_steps,
                    max_epochs=self.max_epochs * self.num_training_steps,
                    warmup_start_lr=self.warmup_start_lr
                    if self.warmup_epochs > 0
                    else self.learning_rate,
                    eta_min=self.final_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }
        else:
            scheduler = None

        return ([optimizer], [scheduler]) if scheduler is not None else optimizer

    def set_optim_algo(self, **kwargs):
        optim_algo = {"Adam": torch.optim.Adam, "Novograd": torch_optimizer.NovoGrad}

        if self.optim_algo not in optim_algo.keys():
            raise ValueError(f"{self.optim_algo} not in {optim_algo.keys()}")

        return optim_algo[self.optim_algo]
