import torch
import torch.nn.functional as F
from pl_bolts.models.self_supervised.ssl_finetuner import SSLFineTuner
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from torchmetrics import Accuracy, MaxMetric, MetricCollection, StatScores, ConfusionMatrix, AUROC


class ExactFineTuner(SSLFineTuner):
    """
    This class implements finetunjng ssl module. It attaches a neural net on top and trains it.
    """
    def __init__(
            self,
            backbone: torch.nn.Module,
            ckpt_path: str,
            semi_sup: bool = False,
            batch_size: int = 32,
            epochs: int = 100,
            **kwargs
    ):
        super(ExactFineTuner, self).__init__(backbone=backbone, **kwargs)

        # whether to do semi supervised or not
        self.semi_sup = semi_sup
        if self.semi_sup:
            self.backbone.train()

        self.warmup_epochs = 10
        self.warmup_start_lr = 0.0
        self._num_training_steps = None
        self.scheduler_interval = "step"
        self.batch_size = batch_size
        self.max_epochs = epochs
        self.num_classes = kwargs['num_classes']

        self.backbone = backbone.load_from_checkpoint(ckpt_path, strict=False)

        # for memorizing all logits
        self.all_val_online_logits = []
        self.all_test_online_logits = []

        # metrics for logging
        metrics = MetricCollection({
            'finetune_acc': Accuracy(num_classes=self.num_classes, multiclass=True),
            'finetune_acc_macro': Accuracy(num_classes=self.num_classes, average='macro', multiclass=True),
            'finetune_auc': AUROC(num_classes=self.num_classes),
            'finetune_stats': StatScores(num_classes=self.num_classes)
        })
        self.train_acc = Accuracy()
        self.val_metrics = metrics.clone(prefix='val/')
        self.test_metrics = metrics.clone(prefix='test/')
        self.val_acc_best = MaxMetric()

    def shared_step(self, batch):
        x, y = batch

        if self.semi_sup:
            feats = self.backbone(x)['feats']
        else:
            with torch.no_grad():
                feats = self.backbone(x)['feats']

        feats = feats.view(feats.size(0), -1)
        logits = self.linear_layer(feats)
        loss = F.cross_entropy(logits, y)

        return loss, logits, y

    def training_step(self, batch, batch_idx):
        loss, logits, y = self.shared_step(batch)
        acc = self.train_acc(logits.softmax(-1), y)

        self.log("train/finetune_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("train/finetune_acc", acc, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx: int):
        loss, logits, y = self.shared_step(batch)
        kwargs = {'on_step': False, 'on_epoch': True, 'sync_dist': True}

        if dataloader_idx == 0:
            self.val_metrics(logits.softmax(-1), y)
            self.all_val_online_logits.append(logits)
            tp, fp, tn, fn, sup = self.val_metrics['val/finetune_stats'].compute()

            self.log_dict(self.val_metrics, prog_bar=True, **kwargs)
            self.log("val/finetune_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/finetune_sen", tp/(tp+fn), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("val/finetune_spe", tn/(tn+fp), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        elif dataloader_idx == 1:
            self.test_metrics(logits.softmax(-1), y)
            self.all_test_online_logits.append(logits)
            tp, fp, tn, fn, sup = self.test_metrics['test/finetune_stats'].compute()

            self.log_dict(self.test_metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/finetune_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/finetune_sen", tp/(tp+fn), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
            self.log("test/finetune_spe", tn/(tn+fp), on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        return loss

    def validation_epoch_end(self, outs):
        all_val_preds = torch.cat(self.all_val_online_logits).argmax(dim=1).detach().cpu().numpy()
        val_targets = self.trainer.datamodule.val_ds.labels[:len(all_val_preds)]

        val_acc = (all_val_preds == val_targets).sum() / len(val_targets)
        self.val_acc_best.update(val_acc)

        self.log("val/finetune_acc_best", self.val_acc_best.compute(), on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        part1 = (all_val_preds[val_targets==0] == val_targets[val_targets==0]).sum()/len(val_targets[val_targets==0])
        part2 = (all_val_preds[val_targets==1] == val_targets[val_targets==1]).sum()/len(val_targets[val_targets==1])
        val_acc_test = (part1+part2)*.5
        self.log("val/finetune_acc_macro-manu", val_acc_test, on_step=False, on_epoch=True,
                 prog_bar=True, sync_dist=True)

    def test_step(self, batch, batch_idx):
        pass

    def on_epoch_end(self):
        # reset all saved logits
        self.all_val_online_logits = []
        self.all_test_online_logits = []
        self.train_acc.reset()
        self.val_metrics.reset()
        self.test_metrics.reset()

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

    def configure_optimizers(self):
        opt_params = [{"params":self.linear_layer.parameters()}, {"params":self.backbone.parameters()}]\
           if self.semi_sup else self.linear_layer.parameters()
        optimizer = torch.optim.Adam(
            opt_params,
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # set scheduler
        if self.scheduler_type == "step":
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, self.decay_epochs, gamma=self.gamma)
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
                    warmup_start_lr=self.warmup_start_lr if self.warmup_epochs > 0 else self.learning_rate,
                    eta_min=self.final_lr,
                ),
                "interval": self.scheduler_interval,
                "frequency": 1,
            }

        return [optimizer], [scheduler]
