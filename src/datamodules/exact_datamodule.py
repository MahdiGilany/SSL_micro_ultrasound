from typing import Optional, Tuple

import torch
from munch import Munch
from pytorch_lightning import LightningDataModule
from torch.utils.data import ConcatDataset, DataLoader, Dataset, random_split
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from .components.data_utils import *
from .components.datasets import *


class ExactDataModule(LightningDataModule):
    """Example of LightningDataModule for Exact dataset.

    A DataModule implements 5 key methods:
        - prepare_data (things to do on 1 GPU/TPU, not on every GPU/TPU in distributed mode)
        - setup (things to do on every accelerator in distributed mode)
        - train_dataloader (the training dataloader)
        - val_dataloader (the validation dataloader(s))
        - test_dataloader (the test dataloader(s))

    This allows you to share a full dataset without explaining how to download,
    split, transform and process the data.

    Read the docs:
        https://pytorch-lightning.readthedocs.io/en/latest/extensions/datamodules.html
    """

    def __init__(
            self,
            data_dir: str = "data/",
            batch_size: int = 32,
            num_workers: int = 0,
            pin_memory: bool = False,
            train_val_split: float = 0.25,
            split_randomstate: int = 26,
            sampler: bool = True,
            dataset_hyp: dict = None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        self.save_hyperparameters(logger=False)

        self.meta_data: dict = {}

        self.lateral_sz = 46.08 # mm
        self.axial_sz = 28 # mm

        self.train_ds: Optional[Dataset] = None
        self.val_ds: Optional[Dataset] = None
        self.test_ds: Optional[Dataset] = None

        # data transformations
        self.transforms = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.,), (1.,))]
        )


    @property
    def num_classes(self) -> int:
        return 2

    def prepare_data(self):
        """load data.

        This method is called only from a single GPU.
        Do not use it to assign state (self.x = y).
        """

        # load meta data
        print("loading meta data...")
        self.meta_data = load_pickle(self.hparams.data_dir+'metadata.pkl')
        print("meta data loaded.")

        # remove empty data from meta data + remove cores with inv less than cutoff_inv
        self.meta_data = remove_empty_lowinv_data(self.meta_data,  dataset_hyp=self.hparams.dataset_hyp)

        # resplit train and val in meta data
        self.meta_data = merge_split_train_val(self.meta_data, random_state=self.hparams.split_randomstate,
                                                train_val_split=self.hparams.train_val_split)

        # finding patch centers on needle region (patches for supervised learning)
        self.patch_centers_sl = estimate_patchcenter(self.meta_data, dataset_hyp=self.hparams.dataset_hyp)

        # find data roots by walking through the data directory
        self.data_roots = get_data_roots(self.hparams.data_dir)

        self.extended_metadata = Munch({'meta_data':self.meta_data, 'patch_centers_sl':self.patch_centers_sl,
                                  'data_roots':self.data_roots})

    def setup(self, stage: Optional[str] = None):
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning when doing `trainer.fit()` and `trainer.test()`,
        so be careful not to execute the random split twice! The `stage` can be used to
        differentiate whether it's called before trainer.fit()` or `trainer.test()`.
        """

        # load datasets only if they're not loaded already
        if not self.train_ds and not self.val_ds and not self.test_ds:
            for state in ['train', 'test', 'val']:
                ds = ExactDataset(state=state, dataset_hyp=self.hparams.dataset_hyp,
                                  extended_metadata=self.extended_metadata)
                self.__setattr__(f'{state}_ds', ds)

    def train_dataloader(self):
        sampler = None
        shuffle = True
        if self.hparams.sampler:
            shuffle = False
            weights = self.balancing_weights()
            weights = torch.Tensor(weights)
            sampler = torch.utils.data.sampler.WeightedRandomSampler(weights, len(weights))

        return DataLoader(
            dataset=self.train_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            sampler=sampler,
        )

    def val_dataloader(self):
        val_loader = DataLoader(
            dataset=self.val_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        test_asval_loader = DataLoader(
            dataset=self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )
        return [val_loader, test_asval_loader]

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def balancing_weights(self,):
        ##todo: code is old and not efficient
        # count for each class -> two classes here
        count = [0] * 2
        dataset = self.train_ds

        count[1] = np.sum(dataset.labels).astype(int)
        count[0] = (len(dataset.labels) - count[1]).astype(int)

        weight_per_class = [0.] * 2
        N = float(sum(count))

        for i in range(2):
            weight_per_class[i] = N / float(count[i])
        weight = [0] * len(dataset.labels)

        for idx, l in enumerate(dataset.labels):
            weight[idx] = weight_per_class[l]
        return weight
