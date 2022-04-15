import torch
from pytorch_lightning.callbacks import Callback


## This module is not used in this project.
class OneCycleLR(Callback):
    def __init__(
            self,
            batch_size,
    ):
        self.batch_sz = batch_size

    def on_fit_start(
            self,
            trainer: "pl.Trainer",
            pl_module: "pl.LightningModule"
    ) -> None:
        # initializing scheduler
        n_epochs = trainer.max_epochs
        len_ds = len(trainer.datamodule.train_ds)
        steps_per_epoch = int(len_ds / self.batch_sz)
        lr = pl_module.hparams.lr
        self.optimizer_ = trainer.optimizers[0]

        self.onecyc_scheduler = \
            torch.optim.lr_scheduler.OneCycleLR(self.optimizer_, float(lr), epochs=n_epochs,
                                                steps_per_epoch=steps_per_epoch,
                                                pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
                                                base_momentum=0.85,
                                                max_momentum=0.95, div_factor=10.0,
                                                final_div_factor=10000.0, three_phase=False,
                                                last_epoch=-1, verbose=False)

    def on_train_batch_end(
        self,
        trainer: "pl.Trainer",
        pl_module: "pl.LightningModule",
        outputs,
        batch,
        batch_idx: int,
        unused: int = 0,
    ) -> None:
        self.onecyc_scheduler.step()
