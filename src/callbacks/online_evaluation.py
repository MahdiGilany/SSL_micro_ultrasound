from audioop import add
from curses import ERR
import os
from pytorch_lightning.callbacks import Callback
from typing import Optional, Sequence
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from torch import nn
import pytorch_lightning as pl
from ..models.typing import FeatureExtractionProtocol
import logging
import torch
from torchmetrics.functional import auroc, accuracy
from pl_bolts.optimizers import LinearWarmupCosineAnnealingLR
from tqdm import tqdm
from pytorch_lightning.loggers import LightningLoggerBase


logger = logging.getLogger("Callbacks.OnlineEvaluation")


def add_prefix_to_dict(d, prefix: str, seperator="/"):
    return {f"{prefix}{seperator}{k}": v for k, v in d.items()}


class EarlyStoppingMonitor:
    def __init__(self, patience, on_early_stop_triggered):
        self.patience = patience
        self.callback_fn = on_early_stop_triggered

        self.strikes = 0
        self.best_score = -1e9

        self.logger = logging.getLogger("Early Stopping")

    def update(self, score):
        if score > self.best_score:
            self.strikes = 0
            self.logger.info(
                f"Registered score of {score} which is higher than previous best {self.best_score}"
            )
            self.best_score = score
        else:
            self.strikes += 1
            if self.strikes >= self.patience:
                self.callback_fn()
                self.logger.info("Early stopping triggered. ")


class ScoreMonitor:
    def __init__(self, mode):
        assert mode in ["min", "max"], f"Only modes `min` and `max` are supported."
        self.mode = mode
        self.best = -1e9 if mode == "max" else 1e9

    def condition(self, old_score, new_score):
        if self.mode == "max":
            return new_score > old_score
        else:
            return new_score < old_score

    def __call__(self, new_value):
        """Updates the value and returns True if this is the best score"""
        if self.condition(self.best, new_value):
            self.best = new_value
            return True
        else:
            return False


class OnlineEvaluation(Callback):
    def __init__(
        self,
        num_classes,
        datamodule: LightningDataModule,
        num_epochs: int = 10,
        evaluate_every_n_epochs: int = 1,
        log_best_only=True,
        reinit_every_epoch=True,
        lr=1e-4,
        weight_decay=1e-6,
        scheduler_epochs=100,
        warmup_epochs=10,
        patience=5,
        checkpoint_monitors: Optional[list[str]] = ["auroc"],
        global_monitor="auroc",
    ):
        self.num_classes = num_classes
        self.datamodule = datamodule
        self.num_epochs = num_epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.scheduler_epochs = scheduler_epochs
        self.warmup_epochs = warmup_epochs
        self.patience = patience
        self.reinit_every_epoch = reinit_every_epoch
        self.evaluate_every_n_epochs = evaluate_every_n_epochs
        self.log_best_only = log_best_only
        self.global_monitor = global_monitor

        self.total_epochs = 0

        self.checkpoint_monitors = checkpoint_monitors
        if self.checkpoint_monitors is not None:
            self.checkpoint_configured = True
            self.score_monitors = {
                name: ScoreMonitor("max") for name in self.checkpoint_monitors
            }
            self.checkpoint_paths = {}
            # TODO - currently we assume mode is max for all monitors

        self.best_train_metrics_global = {}
        self.best_val_metrics_global = {}
        self.best_test_metrics_global = {}
        self.best_val_score_global = -1e9

    def setup(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        stage: Optional[str] = None,
    ) -> None:

        logger.info("Setting up datamodule for online evaluation")
        self.datamodule.setup()
        self.pl_module = pl_module

        assert isinstance(
            self.pl_module, FeatureExtractionProtocol
        ), f"""
            The lightning module used with this callback should implement the <src.models.typing.FeatureExtractionProtocol> interface. 
        """

        self.feature_dim = self.pl_module.features_dim

        self.get_features = self.pl_module.get_features
        self.num_training_steps_per_epoch = len(self.datamodule.train_dataloader())

        self.linear_layer = torch.nn.Linear(self.feature_dim, self.num_classes)

        # attach linear layer reference to pl_module so it gets moved to correct device
        self.pl_module.linear_layer = self.linear_layer

        self.train_loader = self.datamodule.train_dataloader()
        self.val_loader = self.datamodule.val_dataloader()
        if isinstance(self.val_loader, Sequence):
            self.val_loader = self.val_loader[0]
        self.test_loader = self.datamodule.test_dataloader()
        if isinstance(self.test_loader, Sequence):
            self.test_loader = self.test_loader[0]

        assert trainer.logger is not None, f"No logger found for trainer"
        self.logger: LightningLoggerBase = trainer.logger

        if self.checkpoint_configured:
            ERR_MSG = "Trainer must be configured with checkpoint callback to use checkpoint for online trainer"
            assert trainer.checkpoint_callback is not None, ERR_MSG
            assert trainer.checkpoint_callback.dirpath is not None, ERR_MSG
            self.checkpoint_dir = trainer.checkpoint_callback.dirpath

        self.trainer = trainer
        trainer.online_eval_callback = self

    def _build_optimizer(self):
        optimizer = torch.optim.Adam(
            self.linear_layer.parameters(),
            self.lr,
            weight_decay=self.weight_decay,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.warmup_epochs * self.num_training_steps_per_epoch,
            max_epochs=self.scheduler_epochs * self.num_training_steps_per_epoch,
        )

        self.optimizer = optimizer
        self.scheduler = scheduler

    def _trigger_early_stop(self):
        self._should_early_stop = True

    def on_train_epoch_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        assert self.pl_module is not None

        current_pretrain_epoch = pl_module.current_epoch
        if not (current_pretrain_epoch % self.evaluate_every_n_epochs == 0):
            return

        logging.info("Online evaluation Beginning:")

        if self.reinit_every_epoch:
            self.linear_layer.reset_parameters()

        self._build_optimizer()

        best_val_score = -1e9
        best_test_metrics = {}
        best_val_metrics = {}
        best_train_metrics = {}

        self._should_early_stop = False

        early_stopping_monitor = EarlyStoppingMonitor(
            self.patience, self._trigger_early_stop
        )

        for i in range(self.num_epochs):

            if self._should_early_stop:
                break

            # run train, val, test
            train_metrics = self._epoch(
                self.train_loader,
                True,
                state_description=f"Online evaluation epoch {i} - train",
            )
            val_metrics = self._epoch(
                self.val_loader,
                False,
                state_description=f"Online evaluation epoch {i} - val",
            )
            test_metrics = self._epoch(
                self.test_loader,
                False,
                state_description=f"Online evaluation epoch {i} - test",
            )

            # check and update best metrics for this online evaluation routine
            if (s := val_metrics[self.global_monitor]) > best_val_score:
                best_val_score = s
                best_test_metrics = test_metrics
                best_val_metrics = val_metrics
                best_train_metrics = train_metrics

            # check and update best metrics across all online evaluation routines
            if s := val_metrics[self.global_monitor] > self.best_val_score_global:
                self.best_test_metrics_global = s
                self.best_test_metrics_global = test_metrics
                self.best_val_metrics_global = val_metrics
                self.best_train_metrics_global = train_metrics

            # trigger early stopping
            early_stopping_monitor.update(val_metrics["auroc"])

            # log epoch metrics
            if not self.log_best_only:
                self.logger.log_metrics(
                    add_prefix_to_dict(train_metrics, "online_eval/train", "_"),
                    self.total_epochs,
                )
                self.logger.log_metrics(
                    add_prefix_to_dict(val_metrics, "online_eval/val", "_"),
                    self.total_epochs,
                )
                self.logger.log_metrics(
                    add_prefix_to_dict(test_metrics, "online_eval/test", "_"),
                    self.total_epochs,
                )

            # save checkpoint if needed
            if self.checkpoint_configured:
                for name in self.checkpoint_monitors:
                    # updates the score and determines if we should save checkpoint
                    current_score = val_metrics[name]
                    if self.score_monitors[name](current_score):
                        self._checkpoint_model(f"val_{name}", current_score)

            self.total_epochs += 1

        # log best metrics for epoch
        self.logger.log_metrics(
            add_prefix_to_dict(best_train_metrics, "online_eval/best_train", "_"),
            current_pretrain_epoch,
        )
        self.logger.log_metrics(
            add_prefix_to_dict(best_val_metrics, "online_eval/best_val", "_"),
            current_pretrain_epoch,
        )
        self.logger.log_metrics(
            add_prefix_to_dict(best_test_metrics, "online_eval/best_test", "_"),
            current_pretrain_epoch,
        )

        logging.info("Online evaluation complete.")

    def _checkpoint_model(self, monitor_name, monitor_value):
        logger.info("Saving checkpoint to ")
        fpath = os.path.join(
            self.checkpoint_dir,
            f"online_best_{monitor_name}.ckpt",
        )
        self.checkpoint_paths[monitor_name] = fpath
        logger.info(
            f"""
        Saving checkpoint to: 
            {fpath}
        """
        )
        self.trainer.save_checkpoint(fpath)

    def _epoch(self, loader, train=True, state_description="Online Evaluation"):

        assert self.pl_module is not None
        self.pl_module.eval()

        assert self.linear_layer is not None, "Call self._build_linear_layer()"

        optimizer, scheduler = self.optimizer, self.scheduler

        all_logits = []
        all_labels = []

        with tqdm(loader, desc=state_description) as pbar:
            for batch in pbar:

                x, y, metadata = batch
                x = x.to(self.pl_module.device)
                y = y.to(self.pl_module.device)
                with torch.no_grad():
                    features = self.get_features(x)

                logits = self.linear_layer(features)
                loss = nn.functional.cross_entropy(logits, y)

                all_logits.append(logits.detach().cpu())
                all_labels.append(y.detach().cpu())

                if train:
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()

            self.pl_module.log("online_eval_lr", optimizer.param_groups[0]["lr"])
            logits = torch.concat(all_logits, dim=0)
            labels = torch.concat(all_labels, dim=0)

            metrics = self._compute_metrics(logits, labels)
            pbar.set_postfix({"auroc": metrics["auroc"]})

            self.pl_module.log

        return metrics

    def _compute_metrics(self, logits, labels):
        return {
            "auroc": auroc(logits, labels, num_classes=self.num_classes).item(),
            "acc_macro": accuracy(
                logits, labels, average="macro", num_classes=self.num_classes
            ).item(),
        }

    def on_fit_end(
        self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
    ) -> None:
        self.logger.log_metrics(
            add_prefix_to_dict(
                self.best_train_metrics_global, "online_eval/global_best_train", "_"
            )
        )
        self.logger.log_metrics(
            add_prefix_to_dict(
                self.best_val_metrics_global, "online_eval/global_best_val", "_"
            )
        )
        self.logger.log_metrics(
            add_prefix_to_dict(
                self.best_test_metrics_global, "online_eval/global_best_test", "_"
            )
        )
