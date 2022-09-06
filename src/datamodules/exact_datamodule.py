from torch.utils.data import DataLoader
from exactvu.data.datamodule import ExactSSLDataModule
from exactvu.data.datamodule import ConcatenatedCoresDataModule


class ExactDataModule(ExactSSLDataModule):
    def val_dataloader(self):
        val_loader = super(ExactDataModule, self).val_dataloader()
        test_asval_loader = super(ExactDataModule, self).test_dataloader()

        val_loader = val_loader if isinstance(val_loader, list) else [val_loader]
        test_asval_loader = test_asval_loader if isinstance(test_asval_loader, list) else [test_asval_loader]

        return val_loader + test_asval_loader
    def test_dataloader(self):
        return self.val_dataloader()


class ExactCoreDataModule(ConcatenatedCoresDataModule):
    def val_dataloader(self):
        val_loader = super(ExactCoreDataModule, self).val_dataloader()
        test_asval_loader = super(ExactCoreDataModule, self).test_dataloader()

        val_loader = val_loader if isinstance(val_loader, list) else [val_loader]
        test_asval_loader = test_asval_loader if isinstance(test_asval_loader, list) else [test_asval_loader]

        return val_loader + test_asval_loader
