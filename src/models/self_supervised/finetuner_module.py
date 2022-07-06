from typing import Optional, Literal
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
        **kwargs
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

        self.backbone = backbone
        if ckpt_path:
            self.backbone = backbone.load_from_checkpoint(ckpt_path, strict=False)
        else:
            warn(
                "You are using the finetuner model with no loadable checkpoint. The model will be randomly initialized."
            )
            assert semi_sup==True, "checkpoint is not used. Semi-supervised mode must be True"

        # # for memorizing all logits
        # self.all_val_online_logits = []
        # self.all_test_online_logits = []

        # # metrics for logging
        # metrics = MetricCollection({
        #         # 'finetune_acc': Accuracy(num_classes=self.num_classes, multiclass=True),
        #         "finetune_acc_macro": Accuracy(num_classes=self.num_classes, average="macro", multiclass=True),
        #         # 'finetune_auc': AUROC(num_classes=self.num_classes),
        #     })
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
        x, y = batch

        if self.semi_sup:
            feats = self.backbone(x)["feats"]
        else:
            with torch.no_grad():
                feats = self.backbone(x)["feats"]

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
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
        loss, logits, y = self.shared_step(batch)

        self.logging_combined_centers_loss(dataloader_idx, loss)

        return loss, logits, y

    def validation_epoch_end(self, outs):
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True, 'add_dataloader_idx': False}
        self.log("val/finetune_loss", torch.mean(torch.tensor(self.val_macroLoss_all_centers)), prog_bar=True, **kwargs)
        self.log("test/finetune_loss", torch.mean(torch.tensor(self.test_macroLoss_all_centers)), prog_bar=True, **kwargs)

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
                {'params': self.linear_layer.parameters()},
                {'params': self.backbone.parameters()},
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
        optim_algo = {
            'Adam': torch.optim.Adam,
            'Novograd': torch_optimizer.NovoGrad
        }

        if self.optim_algo not in optim_algo.keys():
            raise ValueError(f"{self.optim_algo} not in {optim_algo.keys()}")

        return optim_algo[self.optim_algo]

    def logging_combined_centers_loss(self, dataloader_idx, loss):
        """macro loss for centers"""
        self.inferred_no_centers = dataloader_idx + 1 \
            if dataloader_idx + 1 > self.inferred_no_centers \
            else self.inferred_no_centers

        if dataloader_idx < self.inferred_no_centers/2.:
            self.val_macroLoss_all_centers.append(loss)
        else:
            self.test_macroLoss_all_centers.append(loss)