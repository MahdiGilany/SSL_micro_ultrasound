from torch.utils.data import DataLoader
from exactvu.data.datamodule import ExactSSLDataModule
# from exactvu.data.datamodules import MICCAI2022DataModule


class ExactDataModuleUVA600(ExactSSLDataModule):

    def __init__(
            self,
            *args,
            **kwargs,
    ):
        super().__init__(*args, **kwargs)

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        test_asval_loader = DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        return [val_loader, test_asval_loader]

# class ExactDataModuleUVA600(MICCAI2022DataModule):
#     """Example of LightningDataModule for Exact dataset.
#
#     A DataModule implements 5 key methods:
#         - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
#         - setup (things to do on every accelerator in distributed mode)
#         - train_dataloader (the training dataloader)
#         - val_dataloader (the validation dataloader(s))
#         - test_dataloader (the test dataloader(s))
#
#     This allows you to share a full dataset without explaining how to download,
#     split, transform and process the data.
#
#     Read the docs:
#         https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
#     """
#
#     def __init__(
#             self,
#             *args,
#             **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#
#
#     @property
#     def num_classes(self) -> int:
#         # todo: it should change based on problem
#         return 2
#
#     @property
#     def train_ds(self):
#         return self.train_dataset
#
#     @property
#     def val_ds(self):
#         return self.val_dataset
#
#     @property
#     def test_ds(self):
#         return self.test_dataset
#
#
#     def val_dataloader(self):
#         val_loader = DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             sampler=self.val_dataset.get_sampler()
#             if self.balance_classes_eval
#             else None,
#         )
#         test_asval_loader = DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#             sampler=self.test_dataset.get_sampler()
#             if self.balance_classes_eval
#             else None,
#         )
#         return [val_loader, test_asval_loader]
