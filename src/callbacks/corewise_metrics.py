import numpy as np
from pytorch_lightning.callbacks import Callback


class CorewiseMetrics(Callback):
    def __init__(self, inv_threshold: float = 0.5):
        super().__init__()
        # threshold to consider a predicted involvement as cancer
        self.inv_threshold = inv_threshold

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # for the purpose of name of logging
        self.pl_moduletype = str(type(pl_module))

        # computing corewise metrics and logging.
        corelen_val = trainer.datamodule.val_ds.all_corelen_sl
        corelen_test = trainer.datamodule.test_ds.all_corelen_sl

        # all preds in order
        all_val_preds = pl_module.all_val_preds
        all_test_preds = pl_module.all_test_preds

        # all core labels
        all_val_coretargets = trainer.datamodule.val_ds.core_labels
        all_test_coretargets = trainer.datamodule.test_ds.core_labels

        all_val_corepreds = []
        all_test_corepreds = []

        # find a label for each core in val
        corelen_cumsum = np.cumsum([0] + corelen_val)
        for i, val in enumerate(corelen_cumsum):
            if i == 0 or val>len(all_val_preds):
                continue
            val_minus1 = corelen_cumsum[i-1]
            core_preds = all_val_preds[val_minus1:val]
            all_val_corepreds.append(core_preds.sum()/len(core_preds))

        # find a label for each core in test
        corelen_cumsum = np.cumsum([0] + corelen_test)
        for i, val in enumerate(corelen_cumsum):
            if i == 0 or val>len(all_test_preds):
                continue
            val_minus1 = corelen_cumsum[i-1]
            core_preds = all_test_preds[val_minus1:val]
            all_test_corepreds.append(core_preds.sum()/len(core_preds))

        scores = self.compute_metrics(np.array(all_val_corepreds), all_val_coretargets, state='val', scores={})
        scores = self.compute_metrics(np.array(all_test_corepreds), all_test_coretargets, state='test', scores=scores)

        self.log_scores(scores, pl_module)

    def compute_metrics(self, preds, targets, state, scores={}):
        ind = np.minimum(len(preds), len(targets), dtype='int')
        core_micro_acc = np.sum((preds >= self.inv_threshold) == targets[:ind]) / len(targets[:ind])

        # save differently if SSL is True
        if 'self_supervised' in self.pl_moduletype:
            scores[state + '/' + 'ssl/core-micro'] = core_micro_acc
        else:
            scores[state + '/' + 'acc/core-micro'] = core_micro_acc
        return scores

    def log_scores(self, scores, pl_module):
        for key in scores.keys():
            pl_module.log(key, scores[key], on_epoch=True)
