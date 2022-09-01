from typing import Optional, Literal, List
import torch
import torch.nn.functional as F
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import (
    Accuracy,
    MaxMetric,
    MetricCollection,
    StatScores,
    ConfusionMatrix,
    AUROC,
)

from warnings import warn
import torch_optimizer

from ..supervised.supervised_model import SupervisedModel


class ExactFineTuner(SSLFineTuner):
    """
    This class implements finetunjng ssl module. It attaches a neural net on top and trains it.
    """

    def __init__(
        self,
        backbone: torch.nn.Module,
        ckpt_path: Optional[str] = None,
        optim_algo: Literal["Adam", "Novograd"] = "Adam",
        semi_sup: bool = False,
        batch_size: int = 32,
        epochs: int = 100,
        **kwargs,
    ):
        super(ExactFineTuner, self).__init__(backbone=backbone, **kwargs)

        # whether to do semi supervised or not
        self.semi_sup = semi_sup

        self.warmup_epochs = 10
        self.warmup_start_lr = 0.0
        self._num_training_steps = None
        self.scheduler_interval = "step"
        self.optim_algo = optim_algo
        self.batch_size = batch_size
        self.max_epochs = epochs
        self.num_classes = kwargs["num_classes"]

        if isinstance(backbone, SupervisedModel):
            self.backbone = backbone
            assert (
                backbone.backbone_name == "resnet10"
            ), f"backbones other than resnet10 not supported"
            self.backbone.backbone.fc = torch.nn.Identity()

        else:
            self.backbone = backbone

        if ckpt_path:
            self.backbone = backbone.load_from_checkpoint(ckpt_path, strict=False)
        else:
            warn(
                "You are using the finetuner model with no loadable checkpoint. The model will be randomly initialized."
            )
            assert semi_sup == True, (
                "checkpoint for loading weights is not determined."
                "If you want to train in supervised fashion, then semi-supervised mode must be True"
            )

        self.train_acc = Accuracy()
        self.inferred_no_centers = 1

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

    def on_train_epoch_start(self) -> None:
        """Changing model to eval() mode has to happen at the
        start of every epoch, and should only happen if we are not in semi-supervised mode
        """
        if not self.semi_sup:
            self.backbone.eval()

    def shared_step(self, batch):
        x, y, *metadata = batch

        if self.semi_sup:
            # feats = self.backbone(x, proj=False)["feats"]
            feats = self.get_feats(x)
        else:
            with torch.no_grad():
                # feats = self.backbone(x, proj=False)["feats"]
                feats = self.get_feats(x)

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y, *metadata

    def get_feats(self, x):
        if isinstance(self.backbone, SupervisedModel):
            return self.backbone.backbone(x)
        else:
            return self.backbone(x)["feats"]

    def training_step(self, batch, batch_idx):
        loss, logits, y, *metadata = self.shared_step(batch)
        self.train_acc(logits.softmax(-1), y)

        self.log(
            "train/finetune_loss",
            loss,
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
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        loss, logits, y, *metadata = self.shared_step(batch)

        self.logging_combined_centers_loss(dataloader_idx, loss)

        return loss, logits, y, *metadata

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

        if self._num_training_steps is None:
            len_ds = len(self.trainer.datamodule.train_ds)
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    def configure_optimizers(self):
        opt_params = (
            [
                {"params": self.linear_layer.parameters()},
                {"params": self.backbone.parameters()},
            ]
            if self.semi_sup
            else self.linear_layer.parameters()
        )

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

        return [optimizer], [scheduler]

    def set_optim_algo(self, **kwargs):
        optim_algo = {"Adam": torch.optim.Adam, "Novograd": torch_optimizer.NovoGrad}

        if self.optim_algo not in optim_algo.keys():
            raise ValueError(f"{self.optim_algo} not in {optim_algo.keys()}")

        return optim_algo[self.optim_algo]

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


from src.models.components.backbones import *
import pytorch_lightning as pl


class ExactCoreFineTuner(pl.LightningModule):
    """
    This class implements finetunjng ssl module on cores not patches. It attaches a neural net on top and trains it.
    """

    _SUPPORTED_HEADS = {
        "attention_classifier": attention_classifier,
        "attention_MIL": attention_MIL,
        "simple_aggregation": linear_aggregation,
    }

    def __init__(
        self,
        head_network: str,
        backbone: torch.nn.Module,
        ckpt_path: Optional[str] = None,
        optim_algo: Literal["Adam", "Novograd"] = "Adam",
        semi_sup: bool = False,
        batch_size: int = 5,
        epochs: int = 100,
        in_features: int = 512,
        num_classes: int = 2,
        dropout: float = 0.0,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-6,
        scheduler_type: str = "warmup_cosine",
        decay_epochs: List = [60, 80],
        gamma: float = 0.1,
        final_lr: float = 0.0,
        warmup_epochs=10,
        warmup_start_lr=0.0,
        **kwargs,
    ):
        super(ExactCoreFineTuner, self).__init__()

        head_nn = self._SUPPORTED_HEADS[head_network]
        self.head_network = head_nn(in_features, num_classes)

        self.backbone = backbone
        if ckpt_path:
            self.backbone = backbone.load_from_checkpoint(ckpt_path, strict=False)
        else:
            warn(
                "You are using the finetuner model with no loadable checkpoint. The model will be randomly initialized."
            )
            assert semi_sup == True, (
                "checkpoint for loading weights is not determined."
                "If you want to train in supervised fashion, then semi-supervised mode must be True"
            )

        self.optim_algo = optim_algo
        # whether to do semi supervised or not
        self.semi_sup = semi_sup
        self.batch_size = batch_size
        self.max_epochs = epochs
        self.in_features = in_features
        self.num_classes = num_classes
        self.drop_out = dropout

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.scheduler_type = scheduler_type
        self.decay_epochs = decay_epochs
        self.gamma = gamma
        self.final_lr = final_lr
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self._num_training_steps = None
        self.scheduler_interval = "step"

        self.train_acc = Accuracy()
        self.inferred_no_centers = 1

        self.val_macroLoss_all_centers = []
        self.test_macroLoss_all_centers = []

    def on_train_epoch_start(self) -> None:
        """Changing model to eval() mode has to happen at the
        start of every epoch, and should only happen if we are not in semi-supervised mode
        """
        if not self.semi_sup:
            self.backbone.eval()

    def shared_step(self, batch):
        virtual_batch = 32
        x, pos, _, core_len_cumsum, __, *metadata = batch

        feats = torch.tensor([], device=x.device)
        for i in range(0, x.shape[0], virtual_batch):
            xi = x[i : i + virtual_batch, ...]
            if self.semi_sup:
                feat = self.backbone(xi, proj=False)["feats"]
            else:
                with torch.no_grad():
                    feat = self.backbone(xi, proj=False)["feats"]
            feats = torch.cat((feats, feat), 0)

        feats = feats.view(feats.size(0), -1)
        logits = self.head_network(feats, core_len_cumsum)
        y = torch.tensor(
            [int(grade != "Benign") for grade in metadata[0]["grade"]], device=x.device
        )
        loss = F.cross_entropy(logits, y)

        return loss, logits, y, *metadata

    def training_step(self, batch, batch_idx):
        loss, logits, y, *metadata = self.shared_step(batch)
        self.train_acc(logits.softmax(-1), y)

        self.log(
            "train/finetune_loss",
            loss,
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
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        loss, logits, y, *metadata = self.shared_step(batch)

        self.logging_combined_centers_loss(dataloader_idx, loss)

        return loss, logits, y, *metadata

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

        if self._num_training_steps is None:
            len_ds = len(self.trainer.datamodule.train_ds)
            self._num_training_steps = int(len_ds / self.batch_size) + 1

        return self._num_training_steps

    def configure_optimizers(self):
        opt_params = (
            [
                {"params": self.head_network.parameters()},
                {"params": self.backbone.parameters()},
            ]
            if self.semi_sup
            else self.head_network.parameters()
        )

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

        return [optimizer], [scheduler]

    def set_optim_algo(self, **kwargs):
        optim_algo = {"Adam": torch.optim.Adam, "Novograd": torch_optimizer.NovoGrad}

        if self.optim_algo not in optim_algo.keys():
            raise ValueError(f"{self.optim_algo} not in {optim_algo.keys()}")

        return optim_algo[self.optim_algo]

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
