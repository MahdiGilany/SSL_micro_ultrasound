from abc import ABC, abstractclassmethod, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Union, overload
from xml.etree.ElementPath import get_parent_map
import pytorch_lightning as pl
import os
from typing import Literal, List, Tuple, Optional, cast, Dict
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate

from exactvu.data.splits import (
    HasProstateMaskFilter,
    InvolvementThresholdFilter,
    Splits,
    SplitsConfig,
)
from exactvu.data.utils import data_dir

from .dataset import PatchesDataset, PatchesGroupedByCoreDataset
from .transforms import TensorAugsConfig, Transform, target_transform

from warnings import warn

from omegaconf import ListConfig
from exactvu.data.transforms import TransformV2, TransformConfig, PrebuiltConfigs
from exactvu.data.core import PatchViewConfig
from exactvu.data.splits import SplitsConfig

import logging

log = logging.getLogger(__name__)


def DEFAULT_LABEL_TRANSFORM(label):
    return torch.tensor(label).long()


def DEFAULT_METADATA_TRANSFORM(metadata):
    return metadata["core_specifier"]


@dataclass
class LoaderConfig:
    batch_size: int = 16
    num_workers: int = 0
    balance_classes_train: bool = True
    train_strategy_ddp: bool = False


@dataclass
class ExactPatchDMConfig:

    root: str = field(default_factory=data_dir)
    loader_config: LoaderConfig = LoaderConfig(
        batch_size=16,
        num_workers=8,
    )
    minimum_involvement: float = 0.4
    splits_config: SplitsConfig = SplitsConfig()
    patch_view_config: PatchViewConfig = PatchViewConfig()
    transform_config: TransformConfig = TransformConfig()


def build_dataloader(dataset, targets, config: LoaderConfig, train: bool):
    from .sampler import WeightedDistributedSampler, get_weighted_sampler
    from torch.utils.data import DistributedSampler

    sampler = None

    # Non distributed
    if not config.train_strategy_ddp:

        if train:
            if config.balance_classes_train:
                sampler = get_weighted_sampler(targets)

    # Distributed
    else:
        if train:
            if config.balance_classes_train:
                sampler = WeightedDistributedSampler(
                    dataset,
                    targets,
                )
            else:
                sampler = DistributedSampler(dataset, shuffle=True)
        else:
            sampler = DistributedSampler(dataset, shuffle=False)

    shuffle = sampler is None and train

    return DataLoader(
        dataset,
        config.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=config.num_workers,
    )


class ExactPatchDataModule(pl.LightningDataModule, ABC):
    def __init__(self, config: ExactPatchDMConfig):
        self.config = config
        self.val_ds = None
        self.test_ds = None

    @abstractmethod
    def _make_dataset_impl(self, cores: List[str], train: bool):
        ...

    def _make_dataset(self, cores, train):
        if isinstance(cores, dict):
            return {
                center: self._make_dataset_impl(cores, train)
                for center, cores in cores.items()
            }

        else:
            return self._make_dataset_impl(cores, train)

    def _make_loader(self, dataset, train: bool):
        if isinstance(dataset, dict):
            return [
                self._make_loader(_dataset, train) for name, _dataset in dataset.items()
            ]
        else:
            return build_dataloader(
                dataset, dataset.labels, self.config.loader_config, train
            )

    def _get_splits(self):
        splits = Splits(self.config.splits_config)
        if tau := self.config.minimum_involvement:
            splits.apply_filters(InvolvementThresholdFilter(tau))
        if self.config.patch_view_config.prostate_region_only:
            splits.apply_filters(HasProstateMaskFilter())

        return splits

    def eval_loaders_as_dict(self):
        from itertools import chain

        assert (
            self.val_ds is not None and self.test_ds is not None
        ), "Call setup() first. "

        if isinstance(self.val_ds, dict):
            return {
                name: loader
                for name, loader in zip(
                    chain(
                        map(lambda name: f"val_{name}", self.val_ds.keys()),
                        map(lambda name: f"test_{name}", self.test_ds.keys()),
                    ),
                    chain(self.val_dataloader(), self.test_dataloader()),
                )
            }

        else:
            return {"val": self.val_dataloader(), "test": self.test_dataloader()}


class PatchDataModuleForSupervisedLearning(ExactPatchDataModule):
    def _make_dataset_impl(self, cores: List[str], train: bool):
        from .dataset import PatchesDatasetNew

        return PatchesDatasetNew(
            os.path.join(self.config.root, "cores_dataset"),
            cores,
            self.config.patch_view_config,
            self.train_transform if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self):
        log.info("Setting up datamodule")

        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.config.splits_config.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up pre-processing transforms")
        self.train_transform = TransformV2(self.config.transform_config)

        # eval transform does not use augmentations
        self.eval_transform = TransformV2(
            TransformConfig(
                out_size=self.config.transform_config.out_size,
                norm_config=self.config.transform_config.norm_config,
                tensor_augs_config=None,
                us_augs_config=None,
            )
        )

        log.info("Setting up datasets")
        self.train_ds = self._make_dataset(
            self.splits.get_train(merge_centers=True), train=True
        )
        self.val_ds = self._make_dataset(self.splits.get_val(), train=False)
        self.test_ds = self._make_dataset(self.splits.get_test(), train=False)

    def train_dataloader(self):
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self):
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self):
        return self._make_loader(self.test_ds, False)


class PatchDataModuleForSelfSupervisedLearning(ExactPatchDataModule):
    def _make_dataset_impl(self, cores: List[str], train: bool):
        from .dataset import PatchesDatasetNew

        return PatchesDatasetNew(
            os.path.join(self.config.root, "cores_dataset"),
            cores,
            self.config.patch_view_config,
            self.augmentations,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self):

        log.info("Setting up datamodule")
        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.config.splits_config.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up augmentation transforms")
        from .transforms import MultiTransform, TransformV2

        self.augmentations = MultiTransform(
            TransformV2(self.config.transform_config),
            TransformV2(self.config.transform_config),
        )

        self.train_ds = self._make_dataset(
            self.splits.get_train(merge_centers=True), True
        )

        self.val_ds = self._make_dataset(self.splits.get_val(merge_centers=True), False)

        self.test_ds = self._make_dataset(
            self.splits.get_test(merge_centers=True), False
        )

    def train_dataloader(self):
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self):
        return self._make_loader(self.test_ds, False)


class PatchesConcatenatedFromCoresDataModule(ExactPatchDataModule):
    def __init__(self, config: ExactPatchDMConfig):
        if config.loader_config.batch_size != 1:
            raise ValueError(f"This datamodule only works with batch size 1.")
        super().__init__(config)

    def _make_dataset_impl(self, cores, train):
        from .dataset import PatchesGroupedByCoreDataset

        return PatchesGroupedByCoreDataset(
            os.path.join(self.config.root, "cores_dataset"),
            cores,
            self.config.patch_view_config,
            self.train_transform if train else self.eval_transform,
            DEFAULT_LABEL_TRANSFORM,
            DEFAULT_METADATA_TRANSFORM,
        )

    def setup(self):
        log.info("Setting up datamodule")

        log.info("Setting up cohort splits")
        log.info(f"Using centers {self.config.splits_config.cohort_specifier}")

        self.splits = self._get_splits()

        log.info("Setting up pre-processing transforms")
        self.train_transform = TransformV2(self.config.transform_config)

        # eval transform does not use augmentations
        self.eval_transform = TransformV2(
            TransformConfig(
                out_size=self.config.transform_config.out_size,
                norm_config=self.config.transform_config.norm_config,
                tensor_augs_config=None,
                us_augs_config=None,
            )
        )

        log.info("Setting up datasets")
        self.train_ds = self._make_dataset(
            self.splits.get_train(merge_centers=True), train=True
        )
        self.val_ds = self._make_dataset(self.splits.get_val(), train=False)
        self.test_ds = self._make_dataset(self.splits.get_test(), train=False)

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return self._make_loader(self.train_ds, True)

    def val_dataloader(self) -> EVAL_DATALOADERS:
        return self._make_loader(self.val_ds, False)

    def test_dataloader(self) -> EVAL_DATALOADERS:
        return self._make_loader(self.test_ds, False)


class ExactSSLDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root=None,
        batch_size: int = 16,
        num_workers: int = 8,
        force_redownload=False,
        train_strategy_ddp: bool = False,
        patch_size_pixels=(256, 256),
        mode: Literal["supervised", "self-supervised"] = "supervised",
        return_metadata=False,
        return_labels=True,
        patch_size_mm=(5, 5),
        patch_stride_mm=(1, 1),
        cohort_specifier="UVA600",  # ['UVA', 'CRCEO']
        resample_train_val=True,
        combine_val_sets_for_centers=False,
        train_centers="all",  # ['UVA']
        val_centers="all",  # ['UVA']
        test_centers="all",
        truncate_train_data: Optional[float] = None,
        train_val_ratio=0.25,
        resample_train_val_seed=26,
        needle_region_only=True,
        minimum_involvement=None,
        needle_region_intersection_threshold=0.6,
        prostate_region_only=False,
        prostate_region_intersection_threshold=0.9,
        balance_classes_train=True,
        normalize_mode: Literal["instance", "global"] = "instance",
        normalize_type: Literal["z-score", "min-max"] = "min-max",
        normalize_truncate: bool = True,
        random_phase_shift=False,
        random_phase_distort=False,
        random_phase_distort_strength=0.1,
        random_phase_distort_freq_limit=0.3,
        random_envelope_distort=False,
        random_envelope_distort_strength=0.2,
        random_envelope_distort_freq_limit=0.1,
        random_bandstop=False,
        random_bandstop_width=0.1,
        random_freq_stretch=False,
        random_freq_stretch_range: Union[float, Tuple[float, float]] = 0.98,
        use_augmentations: bool = False,
        aug_prob: float = 0.2,
        random_erasing: bool = True,
        random_horizontal_flip: bool = True,
        random_invert: bool = True,
        random_vertical_flip: bool = True,
        random_affine_translation: Union[float, Tuple[float, float]] = 0.2,
        random_affine_rotation: int = 0,
        random_affine_shear: List[int] = [0, 0, 0, 0],
        random_resized_crop: bool = False,
        random_resized_crop_scale: Union[float, Tuple[float, float]] = 0.7,
    ):
        """DataModule for Exact ultrasound

        PARAMETERS
        ----------
        batch_size: batch size
        num_workers: number of dataloader processes
        force_redownload: If set to True, this will force the re-downloading and re-preprocessing of the cores
            from the image server.
        train_strategy_ddp: If true, datamodule will utilize distributed samplers for dataloading
        patch_size_pixels: desired pixel dimensions of data items X
        mode: whether the datamodule is in self-supervised mode (X, X') or
            supervised mode (returning X, y)
        patch_size_mm: The physical sizes of the patches of RF data X.
        patch_stride_mm: The step sizes from one patch to another (controls degree of patch overlap)
        resample_train_val: Whether to merge and resample the train and validation sets.
        train_val_ratio: if resampling the training and validation sets, the fraction of PATIENTS (not cores) to use in
            the cohort of training vs validation data
        resample_train_val_seed: seed used to resample the training and validation sets.
        needle_region_only: whether to use only the needle region or all the RF image.
        minimum_involvement: the minimum involvement threshold for a cancerous core to be included in the dataset.
            If None, all cores will be included regardless of involvement.
        needle_region_intersection_threshold: The minimum fraction of intersection with the needle region
            for a patch of RF data to be considered 'inside' the needle region
        prostate_region_only: whether to use only the prostate region or all of the RF image.
            WARNING -- Prostate masks are not available for all images.
        prostate_region_intersection_threshold: The minimum fraction of intersection with the prostate region
            for a patch of RF data to be considered 'inside' the prostate region.
        balance_classes_train: whether to balance the dataset by oversampling patches from cancerous cores in training set
        normalize_mode: whether to use instance normalization for each patch or normalization based on
            global dataset statistics
        normalize_type: whether to use z-score or min-max normalization
        normalize_truncate: whether to use truncation to during normalization to keep a high dynamic range
            and ignore outlier pixels

        AUGMENTATION PARAMS
        ------------
        random_erasing: on or off
        random_horizontal_flip: on or off
        random_invert: on or off
        random_vertical_flip: on or off
        random_affine_translation: fractional amount of translation allowed in vertical, horizontal direction
        random_affine_rotation: degree of allowed rotation (default, 0 - no rotation)
        random_affine_shear: x1, x2, y1, y2 - degree range of shear along lateral axis (x1, x2), and axial axis (y_1, y_2)
        random_resized_crop: on or off
        random_resized_crop_scale: range of scales that the image can be cropped to (eg (0.5, 1)) image can be cropped
            down to 1/2 its original size

        YAML CONFIG TEMPLATE
        ----------------

        batch_size: 16
        num_workers: 8
        force_redownload: False
        patch_size_pixels: [256, 256]
        mode: supervised                    # self-supervised
        patch_size_mm: [5, 5]
        patch_stride_mm: [1, 1]
        cohort_specifier: UVA600
        resample_train_val: True
        train_val_ratio: 0.25
        resample_train_val_seed: 26
        needle_region_only: True
        minimum_involvement: null
        needle_region_intersection_threshold: 0.6
        prostate_region_only: False
        prostate_region_intersection_threshold: 0.9
        balance_classes_train: True
        normalize_mode: "instance"
        normalize_type: "min-max"
        normalize_truncate: True
        random_erasing: True
        random_horizontal_flip: True
        random_invert: True
        random_vertical_flip: True
        random_affine_translation: [0.1, 0.1]
        random_affine_rotation: 0
        random_affine_shear: [0, 0, 0, 0]
        random_resized_crop: False
        random_resized_crop_scale: [0.7, 1]

        """

        super().__init__()

        from .utils import data_dir

        self.root = root if root else data_dir()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.mode = mode
        self.return_metadata = return_metadata
        self.return_labels = return_labels

        self.train_strategy_ddp = train_strategy_ddp

        self.cohort_specifier = (
            list(cohort_specifier)
            if isinstance(cohort_specifier, ListConfig)
            else cohort_specifier
        )
        self.resample_train_val = resample_train_val
        self.train_val_ratio = train_val_ratio
        self.resample_train_val_seed = resample_train_val_seed
        self.combine_val_sets_for_centers = combine_val_sets_for_centers
        self.train_centers = train_centers
        self.val_centers = val_centers
        self.test_centers = test_centers
        self.truncate_train_data = truncate_train_data
        self.patch_size_pixels = patch_size_pixels
        self.patch_size_mm = patch_size_mm
        self.patch_stride_mm = patch_stride_mm
        self.needle_region_only = needle_region_only
        self.needle_region_intersection_threshold = needle_region_intersection_threshold
        self.prostate_region_only = prostate_region_only
        self.prostate_region_intersection_threshold = (
            prostate_region_intersection_threshold
        )

        self.minimum_involvement = minimum_involvement
        self.balance_classes_train = balance_classes_train

        self.normalize_mode = normalize_mode
        self.normalize_type = normalize_type
        self.normalize_truncate = normalize_truncate

        self.random_phase_shift = random_phase_shift
        self.random_phase_distort = random_phase_distort
        self.random_phase_distort_strength = random_phase_distort_strength
        self.random_phase_distort_freq_limit = random_phase_distort_freq_limit
        self.random_envelope_distort = random_envelope_distort
        self.random_envelope_distort_strength = random_envelope_distort_strength
        self.random_envelope_distort_freq_limit = random_envelope_distort_freq_limit
        self.random_bandstop = random_bandstop
        self.random_bandstop_width = random_bandstop_width
        self.random_freq_stretch = random_freq_stretch
        self.random_freq_stretch_range = random_freq_stretch_range

        if isinstance(random_freq_stretch_range, float):
            self.random_freq_stretch_range = [random_freq_stretch_range, 1.0]
        else:
            self.random_freq_stretch_range = random_freq_stretch_range

        self.use_augmentations = use_augmentations
        self.aug_prob = aug_prob
        self.random_erasing = random_erasing
        self.random_horizontal_flip = random_horizontal_flip
        self.random_vertical_flip = random_vertical_flip
        self.random_invert = random_invert
        if isinstance(random_affine_translation, float):
            self.random_affine_translation = [
                random_affine_translation,
            ] * 2
        else:
            self.random_affine_translation = random_affine_translation

        self.random_affine_rotation = random_affine_rotation
        self.random_affine_shear = random_affine_shear
        self.random_resized_crop = random_resized_crop
        if isinstance(random_resized_crop_scale, float):
            self.random_resized_crop_scale = [random_resized_crop_scale, 1.0]
        else:
            self.random_resized_crop_scale = random_resized_crop_scale
        self.force_redownload = force_redownload

        self.save_hyperparameters()

        assert mode in ["supervised", "self-supervised"]

        if not needle_region_only and mode == "supervised":
            raise ValueError(
                "Supervised mode does not work with data outside the needle region"
            )

        if self.train_strategy_ddp and self.mode == "supervised":
            warn(
                r"""Using ddp mode with supervised mode may some duplicated elements
            in batches from the end of the dataset. Take care when evaluating the model."""
            )

        self.prepare_data_per_node = False
        self.train_ds, self.val_ds, self.test_ds = None, None, None

    @property
    def num_classes(self):
        return 2

    def get_splits(self):

        from .splits import (
            get_splits,
            filter_splits,
            InvolvementThresholdFilter,
            HasProstateMaskFilter,
        )

        splits = get_splits(
            self.cohort_specifier,
            self.resample_train_val,
            self.resample_train_val_seed,
            self.train_val_ratio,
        )

        # apply filters
        if isinstance(splits, dict):
            if self.prostate_region_only:
                splits = {
                    _center: filter_splits(_splits, HasProstateMaskFilter())
                    for _center, _splits in splits.items()
                }

            if self.minimum_involvement:
                splits = {
                    _center: filter_splits(
                        _splits, InvolvementThresholdFilter(self.minimum_involvement)
                    )
                    for _center, _splits in splits.items()
                }

            # merge training sets
            train = []
            for name, (_train, test, val) in splits.items():
                if self.train_centers == "all" or name in self.train_centers:
                    train.extend(_train)

            # possibly merge validation sets
            if self.combine_val_sets_for_centers:
                val = []
                for name, (_, _val, _) in splits.items():
                    if self.val_centers == "all" or name in self.val_centers:
                        val.extend(_val)
            else:
                val = {
                    name: _val
                    for name, (_, _val, _) in splits.items()
                    if (self.val_centers == "all" or name in self.val_centers)
                }
                self.val_ds_idx2name = [name for name, _ in val.items()]

            # keep test sets separate
            test = {
                name: _test
                for name, (_, _, _test) in splits.items()
                if (self.test_centers == "all" or name in self.test_centers)
            }
            self.test_ds_idx2name = [name for name, _ in test.items()]

            # possibly truncate some of the training data
            if self.truncate_train_data is not None:
                num_train_cores = len(train)
                num_train_cores_to_keep = int(
                    self.truncate_train_data * num_train_cores
                )
                from random import Random

                train = Random(0).sample(train, num_train_cores_to_keep)

            splits = train, val, test

            return splits

        else:
            if self.prostate_region_only:
                splits = filter_splits(splits, HasProstateMaskFilter())

            if self.minimum_involvement:
                splits = filter_splits(
                    splits, InvolvementThresholdFilter(self.minimum_involvement)
                )

            return splits

    def setup(self, stage=None) -> None:
        """
        creates the transforms and dataset classes for the datamodule
        """

        if not self.train_ds and not self.val_ds and not self.test_ds:

            self.train_cores, self.val_cores, self.test_cores = self.get_splits()

            self.train_transform = Transform(
                use_augmentations=True
                if self.mode == "self-supervised"
                else self.use_augmentations,
                create_pairs=self.mode == "self-supervised",
                random_phase_shift=self.random_phase_shift,
                random_phase_distort=self.random_phase_distort,
                random_phase_distort_strength=self.random_phase_distort_strength,
                random_phase_distort_freq_limit=self.random_phase_distort_freq_limit,
                random_envelope_distort=self.random_envelope_distort,
                random_envelope_distort_strength=self.random_envelope_distort_strength,
                random_envelope_distort_freq_limit=self.random_envelope_distort_freq_limit,
                random_bandstop=self.random_bandstop,
                random_bandstop_width=self.random_bandstop_width,
                random_freq_stretch=self.random_freq_stretch,
                random_freq_stretch_range=self.random_freq_stretch_range,
                normalize_mode=self.normalize_mode,  # type:ignore
                normalize_type=self.normalize_type,  # type:ignore
                normalize_truncate=self.normalize_truncate,
                random_erasing=self.random_erasing,
                random_invert=self.random_invert,
                random_horizontal_flip=self.random_horizontal_flip,
                random_vertical_flip=self.random_vertical_flip,
                random_affine_translation=self.random_affine_translation,  # type:ignore
                random_affine_rotation=self.random_affine_rotation,
                random_affine_shear=self.random_affine_shear,
                random_resized_crop=self.random_resized_crop,
                random_resized_crop_scale=self.random_resized_crop_scale,  # type:ignore
                out_size=self.patch_size_pixels,
            )

            self.eval_transform = (
                Transform(
                    use_augmentations=False,
                    create_pairs=False,
                    normalize_mode=self.normalize_mode,  # type:ignore
                    normalize_type=self.normalize_type,  # type:ignore
                    normalize_truncate=self.normalize_truncate,
                    out_size=self.patch_size_pixels,
                )
                if self.mode == "supervised"
                else self.train_transform
            )

            self.train_ds = PatchesDataset(
                self.root,
                self.train_cores,
                self.patch_size_mm,
                self.patch_stride_mm,
                (1, 1),
                self.needle_region_only,
                self.needle_region_intersection_threshold,
                self.prostate_region_only,
                self.prostate_region_intersection_threshold,
                return_labels=self.return_labels,
                return_metadata=self.return_metadata,
                transform=self.train_transform,
                target_transform=target_transform,
                force_redownload=self.force_redownload,
            )

            if isinstance(self.val_cores, dict):
                self.val_ds = {
                    name: PatchesDataset(
                        self.root,
                        cores,
                        self.patch_size_mm,
                        self.patch_stride_mm,
                        (1, 1),
                        self.needle_region_only,
                        self.needle_region_intersection_threshold,
                        self.prostate_region_only,
                        self.prostate_region_intersection_threshold,
                        return_labels=self.return_labels,
                        return_metadata=self.return_metadata,
                        transform=self.eval_transform,
                        target_transform=target_transform,
                        force_redownload=self.force_redownload,
                    )
                    for name, cores in self.val_cores.items()
                }

            else:
                self.val_ds = PatchesDataset(
                    self.root,
                    self.val_cores,
                    self.patch_size_mm,
                    self.patch_stride_mm,
                    (1, 1),
                    self.needle_region_only,
                    self.needle_region_intersection_threshold,
                    self.prostate_region_only,
                    self.prostate_region_intersection_threshold,
                    return_labels=self.return_labels,
                    return_metadata=self.return_metadata,
                    transform=self.eval_transform,
                    target_transform=target_transform,
                    force_redownload=self.force_redownload,
                )

            if isinstance(self.test_cores, dict):
                self.test_ds = {
                    name: PatchesDataset(
                        self.root,
                        cores,
                        self.patch_size_mm,
                        self.patch_stride_mm,
                        (1, 1),
                        self.needle_region_only,
                        self.needle_region_intersection_threshold,
                        self.prostate_region_only,
                        self.prostate_region_intersection_threshold,
                        return_labels=self.return_labels,
                        return_metadata=self.return_metadata,
                        transform=self.eval_transform,
                        target_transform=target_transform,
                        force_redownload=self.force_redownload,
                    )
                    for name, cores in self.test_cores.items()
                }

            else:
                self.test_ds = PatchesDataset(
                    self.root,
                    self.test_cores,
                    self.patch_size_mm,
                    self.patch_stride_mm,
                    (1, 1),
                    self.needle_region_only,
                    self.needle_region_intersection_threshold,
                    self.prostate_region_only,
                    self.prostate_region_intersection_threshold,
                    return_labels=self.return_labels,
                    return_metadata=self.return_metadata,
                    transform=self.eval_transform,
                    target_transform=target_transform,
                    force_redownload=self.force_redownload,
                )

    def train_dataloader(self) -> DataLoader:

        assert self.train_ds is not None, "Call setup() first"

        if self.train_strategy_ddp:
            if self.balance_classes_train:

                from .sampler import WeightedDistributedSampler

                sampler = WeightedDistributedSampler(
                    self.train_ds,
                    self.train_ds.labels,
                    shuffle=True,
                )
            else:
                from torch.utils.data import DistributedSampler

                sampler = DistributedSampler(self.train_ds, shuffle=True)
        else:
            sampler = (
                self.train_ds.get_sampler() if self.balance_classes_train else None
            )

        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False if self.balance_classes_train else True,
            sampler=sampler,
        )

    def _eval_loader(self, dataset: Dataset) -> DataLoader:
        if self.train_strategy_ddp:
            from torch.utils.data import DistributedSampler

            sampler = DistributedSampler(dataset, shuffle=False)
        else:
            sampler = None

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            sampler=sampler,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        assert self.val_ds is not None, "Call setup() first"

        if isinstance(self.val_ds, dict):
            return [self._eval_loader(dataset) for dataset in self.val_ds.values()]

        else:
            return self._eval_loader(self.val_ds)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:

        assert self.test_ds is not None, "Call setup() first"

        if isinstance(self.test_ds, dict):
            return [self._eval_loader(dataset) for dataset in self.test_ds.values()]

        else:
            return self._eval_loader(self.test_ds)

    def clear_data(self) -> None:
        """Clears the from the hard drive"""

        assert (
            self.train_ds is not None
            and self.val_ds is not None
            and self.test_ds is not None
        ), "Call setup() first."

        self.train_ds.clear_data()
        if isinstance(self.val_ds, dict):
            [ds.clear_data() for ds in self.val_ds.values()]
        else:
            self.val_ds.clear_data()

        if isinstance(self.test_ds, dict):
            [ds.clear_data() for ds in self.test_ds.values()]
        else:
            self.test_ds.clear_data()


class ConcatenatedCoresDataModule(pl.LightningDataModule):
    def __init__(
        self,
        root=None,
        batch_size=1,
        num_workers=4,
        return_metadata=False,
        return_labels=True,
        patch_view_config: PatchViewConfig = PatchViewConfig(),
        splits_config: SplitsConfig = SplitsConfig("UVA600"),
        minimum_involvement=None,
        transform_config: Union[TransformConfig, str] = PrebuiltConfigs.BASELINE,
        balance_classes_train=True,
    ):
        super(ConcatenatedCoresDataModule, self).__init__()

        self.root = root if root else data_dir()
        self.batch_size = batch_size
        self.num_workers = num_workers
        if self.batch_size != 1:
            warn("Using this datamodule with batch size greater than 1 is untested.")
        self.return_metadata = return_metadata
        self.return_labels = return_labels
        self.patch_view_config = patch_view_config
        self.splits_config = splits_config
        self.splits_config.cohort_specifier =(
            list(self.splits_config.cohort_specifier)
            if isinstance(self.splits_config.cohort_specifier, ListConfig)
            else self.splits_config.cohort_specifier
        )
        self.minimum_involvement = minimum_involvement
        self.transform_config = (
            transform_config
            if isinstance(transform_config, TransformConfig)
            else cast(TransformConfig, getattr(PrebuiltConfigs, transform_config))
        )
        self.balance_classes_train = balance_classes_train

        self.train_transform = None
        self.eval_transform = None
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    @property
    def cohort_specifier(self):
        return self.splits_config.cohort_specifier

    @overload
    def _make_dataset(
        self, core_specifiers: List[str], train: bool
    ) -> PatchesGroupedByCoreDataset:
        ...

    @overload
    def _make_dataset(
        self, core_specifiers: Mapping[str, List[str]], train: bool
    ) -> Dict[str, PatchesGroupedByCoreDataset]:
        ...

    def _make_dataset(
        self, core_specifiers: Union[List[str], Mapping[str, List[str]]], train: bool
    ):
        from .dataset import PatchesGroupedByCoreDataset

        if isinstance(core_specifiers, list):

            return PatchesGroupedByCoreDataset(
                os.path.join(self.config.root, "cores_dataset"),  # type:ignore
                core_specifiers,
                self.patch_view_config,
                self.return_labels,
                self.return_metadata,
                patch_transform=self.train_transform if train else self.eval_transform,
                target_transform=lambda label: torch.tensor(label).long(),
            )

        else:
            return {
                name: self._make_dataset(_core_specifiers, train)
                for name, _core_specifiers in core_specifiers.items()
            }

    def _get_splits(self):
        from exactvu.data.splits import (
            get_splits,
            filter_splits,
            invert_splits,
            HasProstateMaskFilter,
            InvolvementThresholdFilter,
        )

        splits = get_splits(self.splits_config)

        if self.patch_view_config.prostate_region_only:
            splits = filter_splits(splits, HasProstateMaskFilter())

        if self.minimum_involvement:
            splits = filter_splits(
                splits, InvolvementThresholdFilter(self.minimum_involvement)
            )

        splits = invert_splits(splits)  # type:ignore
        # now have tuple train, val, test of dicts

        # workaround for mergee problem
        train = [vi for v in splits[0].values() for vi in v]

        return train, splits[1], splits[2]
        # return splits

    def setup(self, stage=None):

        if self.train_transform is None:
            self.train_transform = TransformV2(self.transform_config)

        if self.eval_transform is None:
            self.eval_transform = TransformV2(
                TransformConfig(
                    out_size=self.transform_config.out_size,
                    norm_config=self.transform_config.norm_config,
                    tensor_augs_config=None,
                    us_augs_config=None
                )
            )

        train, val, test = self._get_splits()

        self.train_ds = self._make_dataset(train, True)
        self.val_ds = self._make_dataset(val, False)
        self.test_ds = self._make_dataset(test, False)

    def _collate_batch(self, batch):
        patches = [item[0] for item in batch]
        pos = [item[1] for item in batch]

        from .utils import collate_variable_length_tensors

        patches, ind, ptr = collate_variable_length_tensors(patches)
        pos, _, _ = collate_variable_length_tensors(pos)

        from torch.utils.data import default_collate

        remainder = [item[2:] for item in batch]
        remainder = default_collate(remainder)

        def _yield_for_collation():
            yield patches
            yield pos
            yield ind
            yield ptr
            for item in remainder:
                yield item

        return tuple(iter(_yield_for_collation()))

    def _make_dataloader(self, dataset, merge=False, shuffle=False, sampler=None):

        if isinstance(dataset, dict):
            if merge: #todo:this has issues -> workaround is implemented to merge datasets in _get_splits
                return self._make_dataloader(sum(dataset.values()))
            else:
                return [
                    self._make_dataloader(_dataset) for _dataset in dataset.values()
                ]
        else:
            return DataLoader(
                dataset,
                num_workers=self.num_workers,
                batch_size=self.batch_size,
                shuffle=shuffle,
                collate_fn=self._collate_batch,
                sampler=sampler,
            )

    def train_dataloader(self, merge=True):
        assert self.train_ds is not None, "Call `setup` first. "

        if self.balance_classes_train:
            from .sampler import get_weighted_sampler

            if isinstance(self.train_ds, dict):
                train_labels = [labels for ds in self.train_ds.values() for labels in ds.labels]
                sampler = get_weighted_sampler(train_labels)
            else:
                sampler = get_weighted_sampler(self.train_ds.labels)

            return self._make_dataloader(
                self.train_ds, merge=merge, shuffle=False, sampler=sampler
            )

        else:
            return self._make_dataloader(self.train_ds, merge=merge, shuffle=True)

    def val_dataloader(self):
        return self._make_dataloader(self.val_ds)

    def test_dataloader(self):
        return self._make_dataloader(self.test_ds)
