import numpy as np
import torch
from pytorch_lightning.callbacks import Callback

## todo: not useful anymore
class CorewiseMetrics(Callback):
    """
    This module assumes that the pl_module already has pl_module.all_val_online_logits which is all
    the available validation logits.

    Requirements:
        - data module has to have val or test_ds.core_lengths
        - data module has to have val or test_ds.core_labels
        - pl_model has to have all_val_online_logits which contains all logits of the epoch
        - pl_model has to have all_test_online_logits which contains all logits of the epoch

    """
    def __init__(
            self,
            inv_threshold: float = 0.5
    ):
        super().__init__()
        # threshold to consider a predicted involvement as cancer
        self.inv_threshold = inv_threshold

    # todo: all information that is assumed to be available can be obtained in on_validation_batch_end...
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        # for the purpose of name of logging
        self.pl_moduletype = str(type(pl_module))

        # computing corewise metrics and logging.
        corelen_val = trainer.datamodule.val_ds.core_lengths
        corelen_test = trainer.datamodule.test_ds.core_lengths

        # all val and test preds in order
        all_val_logits = torch.cat(pl_module.all_val_online_logits)
        all_test_logits = torch.cat(pl_module.all_test_online_logits)
        all_val_preds = all_val_logits.argmax(dim=1).detach().cpu().numpy()
        all_test_preds = all_test_logits.argmax(dim=1).detach().cpu().numpy()

        # all core labels
        all_val_coretargets = trainer.datamodule.val_ds.core_labels
        all_test_coretargets = trainer.datamodule.test_ds.core_labels

        # find a label for each core in val and test
        all_val_corepreds = self.get_core_preds(all_val_preds, corelen_val)
        all_test_corepreds = self.get_core_preds(all_test_preds, corelen_test)


        # scores is a dict containing all core-wise metrics
        scores = self.compute_metrics(np.array(all_val_corepreds), all_val_coretargets, state='val', scores={})
        scores = self.compute_metrics(np.array(all_test_corepreds), all_test_coretargets, state='test', scores=scores)

        self.log_scores(scores, pl_module)

    def get_core_preds(self, all_val_preds, corelen_val):
        """This function takes the mean of all patches inside a core as prediction of that core."""
        all_val_corepreds = []
        corelen_cumsum = np.cumsum([0] + corelen_val)

        for i, val in enumerate(corelen_cumsum):
            if i == 0 or val > len(all_val_preds):
                continue

            val_minus1 = corelen_cumsum[i-1]
            core_preds = all_val_preds[val_minus1:val]
            all_val_corepreds.append(core_preds.sum()/len(core_preds))

        return all_val_corepreds

    def compute_metrics(self, preds, targets, state, scores={}):
        ind = np.minimum(len(preds), len(targets), dtype='int')
        core_micro_acc = np.sum((preds >= self.inv_threshold) == targets[:ind]) / len(targets[:ind])

        # save differently if SSL is True
        if 'finetune' in self.pl_moduletype: # todo change it soon
            scores[state + '/' + 'finetune_core-micro'] = core_micro_acc
        elif 'self_supervised' in self.pl_moduletype:
            scores[state + '/' + 'ssl/core-micro'] = core_micro_acc
        else:
            scores[state + '/' + 'acc/core-micro'] = core_micro_acc
        return scores

    def log_scores(self, scores, pl_module):
        for key in scores.keys():
            pl_module.log(key, scores[key], on_epoch=True)
