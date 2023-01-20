from collections import namedtuple
from ctypes import pointer
from itertools import chain
from typing import (
    Dict,
    List,
    Optional,
    Callable,
    Any,
    Tuple,
)
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import os
from .core import Core, PatchView, PatchViewConfig
from torch.utils.data.sampler import WeightedRandomSampler
from torch.utils.data import DataLoader

from ..preprocessing import DEFAULT_PREPROCESS_TRANSFORM
from abc import ABC, abstractmethod
from .core import Core
import pickle


class CoresMultiItemDataset(Dataset, ABC):
    def __init__(self, root, core_specifier_list):

        self.root = root

        self.directory = os.path.join(root, "cores_dataset")
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        from exactvu import resources

        metadata = resources.metadata()

        self.metadata = metadata.loc[
            metadata["core_specifier"].isin(core_specifier_list)
        ].reset_index(drop=True)

        self.core_specifiers = tuple(self.metadata["core_specifier"])

        self.cores = [
            Core(core_specifier, self.directory)
            for core_specifier in self.core_specifiers
        ]

        self.core_data = []
        self.core_lengths = []

        for core in tqdm(self.cores):

            self.download_and_preprocess_core(core)
            items = self.get_data_items_for_core(core)
            self.core_data.append(items)
            self.core_lengths.append(len(items))

    @abstractmethod
    def download_and_preprocess_core(self, core: Core) -> None:
        """
        Any processing that needs to be done to the individual core objects
        before the method get_data_items_for_core can be invoked.
        """

    @abstractmethod
    def get_data_items_for_core(self, core: Core) -> Any:
        """Return an object for accessing the data items for the specified core.
        the object should implement the __getitem__ and __len__ protocols
        """

    @abstractmethod
    def transform_single_item(self, item, core_idx, patch_idx, absolute_idx) -> Any:
        """
        The final layer of preprocessing between the data item and the input to the
        network. Could include adding the label for the core to be paired with the
        data item (X, y). Also should include mapping the raw data (np array) to tensor, etc.
        """

    @property
    def core_labels(self):
        return tuple(self.metadata["grade"] != "Benign")

    @property
    def core_inv(self):
        return tuple(self.metadata["pct_cancer"].replace(np.nan, 0))

    @property
    def labels(self):
        labels = []
        for core_idx in range(len(self.cores)):
            labels.extend(
                [
                    self.get_core_metadata(core_idx)["grade"] != "Benign",
                ]
                * self.core_lengths[core_idx]
            )
        return labels

    def get_core_and_patch_idx(self, idx) -> Tuple[int, int]:
        """
        based on the specified index, finds the correct combination of
        core index in dataset, patch index in core.
        """
        total_length = sum(self.core_lengths)
        if idx < 0 or idx >= total_length:
            raise IndexError(
                f"Index {idx} out of bounds for core lengths {self.core_lengths}"
            )

        counter = idx
        patch_idx = 0
        for length in self.core_lengths:
            if counter < length:
                return patch_idx, counter
            else:
                counter -= length
                patch_idx += 1

        raise IndexError

    def get_patch_indices_for_core(self, core_idx: int):
        """Lists the patch indices corresponding to the specified core index

        Args:
            core_idx (int): core index (corresponds to index of metadata table)
        """
        assert self.core_lengths is not None, "Call setup methods first"

        total_lengths = np.cumsum([0, *self.core_lengths])
        return list(range(total_lengths[core_idx], total_lengths[core_idx + 1]))

    def clear_data(self):
        for core in self.cores:
            core.clear_data()

    def get_item_metadata(self, idx) -> Dict[str, Any]:
        core, item_idx = self.get_core_and_patch_idx(idx)
        info = dict(self.metadata.iloc[core])
        info["core_index"] = core  # type:ignore
        info["item_index"] = item_idx  # type:ignore
        return info

    def get_core_metadata(self, core_idx):
        return dict(self.metadata.iloc[core_idx])  # type:ignore

    def get_sampler(self):
        """returns a sampler for this dataset which upsamples cores which are positive for cancer"""

        labels = np.array(self.labels)

        positives = np.sum(labels == True)
        negatives = np.sum(labels == False)
        total = len(labels)
        positives_weight = total / positives
        negatives_weight = total / negatives

        weights = torch.tensor(np.where(labels, positives_weight, negatives_weight))

        return WeightedRandomSampler(weights, len(weights))  # type:ignore

    def __len__(self):
        return sum(self.core_lengths)

    def __getitem__(self, idx):

        if type(idx) == int:

            core_idx, patch_idx = self.get_core_and_patch_idx(idx)
            item = self.core_data[core_idx][patch_idx]
            return self.transform_single_item(item, core_idx, patch_idx, idx)

        else:
            idx = list(idx)
            return [self.__getitem__(_idx) for _idx in idx]


class PatchesDataset(CoresMultiItemDataset):
    def __init__(
        self,
        root: str,
        core_specifier_list: List[str],
        patch_size=(5, 5),
        patch_strides=(1, 1),
        subpatch_size_mm=(1, 1),
        needle_region_only=False,
        needle_intersection_threshold=0.6,
        prostate_region_only=False,
        prostate_intersection_threshold=0.8,
        return_labels=True,
        return_metadata=False,
        return_core_idx=False,
        iq_preprocess_transform: Callable[
            ..., np.ndarray
        ] = DEFAULT_PREPROCESS_TRANSFORM,
        transform: Optional[Callable[[np.ndarray], Any]] = None,
        target_transform: Optional[Callable[[bool], torch.Tensor]] = None,
        force_redownload=False,
    ):

        self.root = root
        self.directory = os.path.join(root, "cores_dataset")
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.core_specifier_list = core_specifier_list
        self.patch_size = patch_size
        self.patch_strides = patch_strides
        self.subpatch_size_mm = subpatch_size_mm
        self.needle_region_only = needle_region_only
        self.needle_intersection_threshold = needle_intersection_threshold
        self.prostate_region_only = prostate_region_only
        self.prostate_intersection_threshold = prostate_intersection_threshold
        self.return_labels = return_labels
        self.return_metadata = return_metadata
        self.return_core_idx = return_core_idx
        self.iq_preprocess_transform = iq_preprocess_transform
        self.transform = transform
        self.target_transform = target_transform
        self.force_redownload = force_redownload

        super(PatchesDataset, self).__init__(root, core_specifier_list)

    def get_data_items_for_core(self, core: Core) -> Any:
        return core.get_patch_view(
            self.patch_size,
            self.patch_strides,
            self.subpatch_size_mm,
            self.needle_region_only,
            self.prostate_region_only,
            self.prostate_intersection_threshold,
            self.needle_intersection_threshold,
        )

    def download_and_preprocess_core(self, core: Core) -> None:
        if (not core.check_is_downloaded()) or self.force_redownload:
            core.download_and_preprocess_iq(self.iq_preprocess_transform)
        if self.prostate_region_only and not core.check_prostate_mask_is_downloaded():
            try:
                core.download_prostate_mask()
            except:
                raise ValueError(
                    f"prostate mask not available for core {core.specifier}."
                )

    def transform_single_item(self, item, core_idx, item_idx, absolute_idx) -> Any:

        item = item[0]
        if self.transform:
            item = self.transform(item)

        label = self.get_core_metadata(core_idx)["grade"] != "Benign"
        if self.target_transform:
            label = self.target_transform(label)

        out = [item]

        if self.return_labels:
            out.append(label)

        if self.return_metadata:
            out.append(self.get_item_metadata(absolute_idx))

        return out[0] if len(out) == 1 else tuple(out)

        # if self.return_labels:
        #    return item, label
        # else:
        #    return item

    def get_item_metadata(self, idx) -> Dict[str, Any]:
        out = super().get_item_metadata(idx)
        core_idx, item_idx = self.get_core_and_patch_idx(idx)
        patch_loc = self.core_data[core_idx].patch_positions[item_idx]
        out["coordinates"] = patch_loc
        return out


class CoresDataset(Dataset, ABC):
    def __init__(self, root, core_specifier_list) -> None:
        super().__init__()

        self.root = root

        from exactvu import resources

        metadata = resources.metadata()
        self.metadata = metadata.loc[
            metadata["core_specifier"].isin(core_specifier_list)
        ].reset_index(drop=True)

        self.core_specifiers = tuple(self.metadata["core_specifier"])

        self.__cores = []
        for specifier in tqdm(self.core_specifiers, desc="Preparing cores"):
            core = Core(specifier, self.root)
            self.prepare_core(core)
            if self.filter_core(core):
                self.__cores.append(core)

    @abstractmethod
    def prepare_core(self, core: Core):
        ...

    def filter_core(self, core: Core):
        return True

    def get_metadata(self, core_idx: int):
        return dict(self.metadata.iloc[core_idx])  # type:ignore

    @property
    def cores(self) -> List[Core]:
        return self.__cores


class ItemsWithSubItemsMixin(object):
    def __init__(self, item_lengths):
        self.item_lengths = item_lengths

    def get_item_and_subitem_idx(self, absolute_idx) -> Tuple[int, int]:
        """
        based on the specified absolute index in the list, returns
        the corresponding tuple of item and subitem indices,
        in the format
            item_idx, subitem_idx
        """
        total_length = sum(self.item_lengths)
        if absolute_idx < 0 or absolute_idx >= total_length:
            raise IndexError(
                f"Index {absolute_idx} out of bounds for core lengths {self.item_lengths}"
            )

        counter = absolute_idx
        item_idx = 0
        for length in self.item_lengths:
            if counter < length:
                return item_idx, counter
            else:
                counter -= length
                item_idx += 1

        raise IndexError

    def get_indices_for_item(self, item_idx):
        if item_idx not in range(len(self.item_lengths)):
            raise IndexError(
                f"Index {item_idx} out of bounds. Total of {len(self.item_lengths)} items in dataset"
            )

        total_lengths = np.cumsum([0, *self.item_lengths])
        return list(range(total_lengths[item_idx], total_lengths[item_idx + 1]))

    def __len__(self):
        return sum(self.item_lengths)


# class ConcatenatedPatchEmbeddingsDataset(CoresDataset):
#    def __init__(
#        self,
#        root,
#        core_specifier_list,
#        patch_transform: Callable[[np.ndarray], torch.Tensor],
#        feature_extractor: torch.nn.Module,
#        compute_device="cuda:0",
#        needle_intersection_threshold=0.66,
#        cache_mode="memory",
#        max_batch_size_for_embeddings_calculation=32,
#        return_metadata=False,
#    ):
#        super().__init__(root, core_specifier_list)
#        self.cache_mode = cache_mode
#        self.patch_transform = patch_transform
#        self.feature_extractor = feature_extractor.to(compute_device)
#        self._cache_hit = [False] * len(self.cores)
#        self.needle_intersection_threshold = needle_intersection_threshold
#        self.compute_device = compute_device
#        self.max_batch_size = max_batch_size_for_embeddings_calculation
#        self.return_metadata = return_metadata
#
#    def prepare_core(self, core: Core):
#        if core.image is None:
#            core.download_and_preprocess_iq()
#
#    def _cache(self, core, item):
#        if self.cache_mode == "memory":
#            core.item = item
#        elif self.cache_mode == "disk":
#            fname = f"{self.__class__.__name__}_cached_item.pkl"
#            fname = os.path.join(core.directory, fname)
#            with open(fname, "wb") as f:
#                pickle.dump(item, f)
#
#    def _retrieve_cache(self, core):
#        if self.cache_mode == "memory":
#            return core.item
#
#        elif self.cache_mode == "disk":
#            fname = f"{self.__class__.__name__}_cached_item.pkl"
#            fname = os.path.join(core.directory, fname)
#            with open(fname, "rb") as f:
#                return pickle.load(f)
#
#    def _get_possibly_cached_item(self, idx):
#
#        core = self.cores[idx]
#
#        if self._cache_hit[idx]:
#            return self._retrieve_cache(core)
#
#        else:
#            core = self.cores[idx]
#            patch_view = core.get_patch_view(
#                needle_region_only=True,
#                needle_intersection_threshold=self.needle_intersection_threshold,
#            )
#            patches = torch.stack(
#                [self.patch_transform(patch) for patch in patch_view]
#            ).to(self.compute_device)
#
#            from ..utils import compute_batched
#
#            def fn(batch):
#                batch = batch.to(self.compute_device)
#                with torch.no_grad():
#                    return self.feature_extractor(batch).to("cpu")
#
#            features = compute_batched(
#                patches,
#                fn,
#            )
#
#            self._cache(core, features)
#            self._cache_hit[idx] = True
#
#            return features
#
#    def __getitem__(self, idx):
#
#        core = self.cores[idx]
#        item = self._get_possibly_cached_item(idx)
#        label = core.metadata["grade"] != "Benign"
#        label = torch.tensor(label).long()
#        metadata = self.get_metadata(idx)
#
#        return (item, label, metadata) if self.return_metadata else (item, label)
#
#    def __len__(self):
#        return len(self.cores)


PatchesDatasetOutput = namedtuple(
    "PatchesDatasetOutput", ["patch", "pos", "label", "metadata"]
)


class PatchesDatasetNew(ItemsWithSubItemsMixin, CoresDataset):
    def __init__(
        self,
        root,
        core_specifier_list,
        patch_view_config: PatchViewConfig = PatchViewConfig(),
        patch_transform=None,
        label_transform=None,
        metadata_transform=None,
    ):

        self.patch_view_config = patch_view_config
        self.patch_transform = patch_transform
        self.label_transform = label_transform
        self.metadata_transform = metadata_transform
        CoresDataset.__init__(self, root, core_specifier_list)

        self.patch_views = [
            core.get_patch_view(self.patch_view_config)
            for core in tqdm(self.cores, desc="Indexing Patches")
        ]
        self.patch_views: List[PatchView]
        self.item_lengths = [len(view) for view in self.patch_views]

        self.labels = list(
            chain(
                *[
                    [self.get_metadata(core_idx)["grade"] != "Benign"]
                    * self.item_lengths[core_idx]
                    for core_idx in range(len(self.cores))
                ]
            )
        )

    def __getitem__(self, idx):

        core_idx, item_idx_in_core = self.get_item_and_subitem_idx(idx)

        patch, position = self.patch_views[core_idx][item_idx_in_core]
        position = np.array(position)

        if self.patch_transform is not None:
            patch = self.patch_transform(patch)

        label = self.get_metadata(core_idx)["grade"] != "Benign"
        if self.label_transform:
            label = self.label_transform(label)

        metadata = self.get_metadata(core_idx)
        if self.metadata_transform:
            metadata = self.metadata_transform(metadata)

        return PatchesDatasetOutput(patch, position, label, metadata)

    def get_metadata(self, core_idx):
        return super().get_metadata(core_idx)

    def prepare_core(self, core: Core):
        
        if not core.check_is_downloaded():
            core.download_and_preprocess_iq()

        if (
            self.patch_view_config.prostate_region_only
            and not core.check_prostate_mask_is_downloaded()
        ):
            success = core.download_prostate_mask()
            if not success:
                raise RuntimeError(
                    f"failed to download prostate mask for core {core.specifier}."
                )


class PatchFeaturesDataset(ItemsWithSubItemsMixin, CoresDataset):
    def __init__(
        self,
        root,
        core_specifiers,
        patch_transform,
        encoder,
        compute_device=None,
        patch_size=(5, 5),
        subpatch_size_mm=(1, 1),
        needle_region_intersection_threshold=0.66,
        prostate_region_intersection_threshold=0.9,
        return_labels=True,
        return_metadata=False,
    ):

        self.compute_device = compute_device
        self.encoder = encoder.to(self.compute_device)
        self.return_labels = return_labels
        self.return_metadata = return_metadata

        CoresDataset.__init__(self, root, core_specifiers)

        self.patches_dataset = PatchesDataset(
            self.root,
            core_specifiers,
            patch_size=patch_size,
            subpatch_size_mm=subpatch_size_mm,
            needle_region_only=needle_region_intersection_threshold > 0,
            needle_intersection_threshold=needle_region_intersection_threshold,
            prostate_region_only=prostate_region_intersection_threshold > 0,
            prostate_intersection_threshold=prostate_region_intersection_threshold,
            return_metadata=True,
            return_labels=False,
            transform=patch_transform,
        )

        self._features_for_cores = {}

        from ..utils import find_largest_batch_size

        shape = self.patches_dataset[0][0].shape
        batch_size = find_largest_batch_size(
            self._compute_features,
            shape,
        )
        loader = DataLoader(self.patches_dataset, batch_size)
        for batch in tqdm(loader, desc="Computing features"):
            self._process_batch(batch)

        item_lengths = []
        for core in self.cores:
            item_lengths.append(len(self._features_for_cores[core.specifier]))

        ItemsWithSubItemsMixin.__init__(self, item_lengths)

    def _compute_features(self, X):
        self.encoder.eval()
        with torch.no_grad():
            return self.encoder(X).detach().cpu()

    def _process_batch(self, batch):
        patches, metadata = batch
        specifiers = metadata["core_specifier"]
        x, y = metadata["coordinates"]
        patches = patches.to(self.compute_device)

        features = self._compute_features(patches)

        for feature, specifier, x_, y_ in zip(features, specifiers, x, y):
            self._add_feature_for_core(feature, specifier, (x_.item(), y_.item()))

    def _add_feature_for_core(self, feature: torch.Tensor, core_specifier, coords):
        self.features_for_core = self._features_for_cores.setdefault(core_specifier, {})
        self.features_for_core[coords] = feature

    def prepare_core(self, core: Core):
        core.download_and_preprocess_iq()

    def get_metadata_for_item(self, idx):
        core_idx, patch_idx = self.get_item_and_subitem_idx(idx)
        out = self.get_metadata(core_idx)
        out["coords"] = self._features_for_cores[out["core_specifier"]].keys()[
            patch_idx
        ]

    def get_feature(self, core_specifier, x, y):
        try:
            feats = self._features_for_cores[core_specifier]
        except:
            raise KeyError(f"No features for core {core_specifier}.")
        try:
            return feats[x, y]
        except:
            raise KeyError(
                f"No features avaible for position {x, y} of core {core_specifier}"
            )

    def __getitem__(self, idx):

        core_idx, item_idx = self.get_item_and_subitem_idx(idx)
        metadata = self.get_metadata(core_idx)
        core_specifier = metadata["core_specifier"]
        label = metadata["grade"] != "Benign"

        feature = self._features_for_cores[core_specifier].values()[item_idx]

        out = [feature]

        if self.return_labels:
            out.append(torch.tensor(label).long())

        if self.return_metadata:
            out.append(self.get_metadata_for_item(idx))

        return out[0] if len(out) == 1 else tuple(out)


from .core import PatchViewConfig


class PatchesGroupedByCoreDataset(CoresDataset):
    def __init__(
        self,
        root: str,
        core_specifier_list: List[str],
        patch_view_config: PatchViewConfig,
        return_labels=True,
        return_metadata=False,
        patch_transform: Optional[Callable[[np.ndarray], Any]] = None,
        target_transform: Optional[Callable[[bool], torch.Tensor]] = None,
    ):

        self.patch_view_config = patch_view_config
        self.return_labels = return_labels
        self.return_metadata = return_metadata
        self.patch_transform = patch_transform
        self.target_transform = target_transform

        super().__init__(root, core_specifier_list)

    @property
    def labels(self):
        return [self.get_metadata(idx)["grade"] != "Benign" for idx in range(len(self))]

    def prepare_core(self, core: Core):
        if core.image is None:
            core.download_and_preprocess_iq()

    def filter_core(self, core: Core):
        return len(core.get_patch_view(self.patch_view_config)) > 0

    def __len__(self):
        return len(self.cores)

    def __getitem__(self, idx):
        core = self.cores[idx]

        patch_view = core.get_patch_view(self.patch_view_config)

        patches = []
        positions = []

        for patch, position in zip(
            patch_view, patch_view.patch_positions  # type:ignore
        ):

            if self.patch_transform:
                patch = self.patch_transform(patch)

            patches.append(patch)
            positions.append(torch.tensor([*position]))

        patches = (
            torch.stack(patches, dim=0)
            if isinstance(patch, torch.Tensor)
            else np.stack(patches, axis=0)
        )

        positions = torch.stack(positions, dim=0)

        out = [patches, positions]

        if self.return_labels:
            label = self.get_metadata(idx)["grade"] != "Benign"
            out.append(self.target_transform(label) if self.target_transform is not None else label)

        if self.return_metadata:
            out.append(self.get_metadata(idx))

        return tuple(out)
