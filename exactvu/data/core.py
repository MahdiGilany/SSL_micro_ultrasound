from dataclasses import asdict, dataclass
from typing import Iterable, Literal, Tuple, overload, Optional
from skimage.transform import resize

from ..preprocessing import (
    patch_size_mm_to_pixels,
    split_into_patches,
    DEFAULT_PREPROCESS_TRANSFORM,
    split_into_patches_pixels,
)
from .grid import InMemoryImagePatchGrid, SavedSubPatchGrid, SubPatchAccessorMixin
from itertools import product
import numpy as np
import os
from ..client import load_by_core_specifier, load_prostate_mask
from exactvu.preprocessing import to_bmode
from enum import Enum, auto

from collections import OrderedDict


def mask_intersections_for_grid(img_shape, mask, subpatch_size_mm=(1, 1)):

    mask = resize(mask.astype("bool"), (img_shape[0], img_shape[1]))
    intersections = split_into_patches(mask, *subpatch_size_mm)
    patch_size = intersections[0, 0].size
    return intersections.sum(axis=(-1, -2)) / patch_size


def mask_intersections_for_rf_slices(mask, rf_slices, rf_shape):
    mask = resize(
        mask.astype("bool"),
        rf_shape,
    )
    intersections = np.zeros(rf_slices.shape[:-1])
    indices = product(*map(range, intersections.shape))
    for idx in indices:
        x1, x2, y1, y2 = rf_slices[idx]
        mask_slice = mask[x1:x2, y1:y2]
        intersection_ratio = mask_slice.sum() / mask_slice.size
        intersections[idx] = intersection_ratio

    return intersections


def grid_slices_to_rf_slices(
    grid_slices: np.ndarray, rf_shape, subpatch_size_mm=(1, 1)
):
    """
    Converts the given array of grid positions corresponding to slices of a base patch grid
    of shape specified by `subpatch_shape` to an array of slice positions corresponding to the
    rf image with the given shape
    """
    patch_size = patch_size_mm_to_pixels(rf_shape, *subpatch_size_mm)

    out = np.zeros_like(grid_slices)
    out[..., 0] = grid_slices[..., 0] * patch_size[0]
    out[..., 1] = grid_slices[..., 1] * patch_size[0]
    out[..., 2] = grid_slices[..., 2] * patch_size[1]
    out[..., 3] = grid_slices[..., 3] * patch_size[1]

    return out


def is_inside_mask(
    x1, x2, y1, y2, mask, subpatch_size_pixels, intersection_threshold=0.6
):
    """check if the patch specified by grid coordinates x1, x2, y1, y2 is inside the given mask,
    assuming the mask is divided into a grid of subpatches with size subpatch_size_pixels.

    The criterion for being inside the mask is an overlap of at least intersection_threshold
    with the mask."""

    intersections = split_into_patches_pixels(mask, *subpatch_size_pixels)
    patch_size = intersections[0, 0].size
    intersections = intersections.sum(axis=(1, -2)) / patch_size

    intersection = needle_intersections_grid[  # type:ignore
        x1:x2, y1:y2
    ]
    intersection = intersection.mean()

    return intersection >= intersection_threshold


class PatchView:

    from collections import namedtuple

    Output = namedtuple("PatchViewOutput", ["patch", "pos"])

    def __init__(self, base_grid, patch_positions):
        self.base_grid = base_grid
        self.patch_positions = patch_positions

    def __len__(self):
        return len(self.patch_positions)

    def __getitem__(self, idx):
        x1, x2, y1, y2 = self.patch_positions[idx]
        patch = self.base_grid[x1:x2, y1:y2]
        return PatchView.Output(patch, (x1, x2, y1, y2))

    @property
    def patch_mask(self):
        msk = np.zeros(self.base_grid.shape)
        for x1, x2, y1, y2 in self.patch_positions:
            msk[x1:x2, y1:y2] = 1
        return msk


class PatchBackends(Enum):
    INDIVIDUAL_SAVED_SUBPATCHES = auto()
    IN_MEMORY_IMAGE = auto()


@dataclass
class PatchViewConfig:
    patch_size: Tuple[int, int] = (5, 5)
    patch_strides: Tuple[int, int] = (1, 1)
    subpatch_size: Tuple[int, int] = (1, 1)
    needle_region_only: Optional[bool] = None
    prostate_region_only: Optional[bool] = None
    prostate_intersection_threshold: float = 0.6
    needle_intersection_threshold: float = 0.6

    def __post_init__(self):
        if self.prostate_region_only is None:
            self.prostate_region_only = self.prostate_intersection_threshold != 0.0
        if self.needle_region_only is None:
            self.needle_region_only = self.needle_intersection_threshold != 0.0


class Core:
    def __init__(
        self,
        specifier,
        root=None,
        backend: PatchBackends = PatchBackends.IN_MEMORY_IMAGE,
    ):

        if root is None:
            from .utils import data_dir

            root = os.path.join(data_dir(), "cores_dataset")
            if not os.path.isdir(root):
                os.mkdir(root)

        self.directory = os.path.join(root, specifier)
        if not os.path.isdir(self.directory):
            os.mkdir(self.directory)

        self.specifier = specifier
        self._metadata = None
        self.backend = backend

    @property
    def image(self):
        if not self.check_is_downloaded():
            return None

        return np.load(os.path.join(self.directory, "image.npy"))

    def check_is_downloaded(self):
        return os.path.isfile(os.path.join(self.directory, "image.npy"))

    @image.setter
    def image(self, image):
        np.save(os.path.join(self.directory, "image.npy"), image)

    @property
    def prostate_mask(self):
        if os.path.isfile(path := os.path.join(self.directory, "prostate_mask.npy")):
            return np.load(path)
        else:
            return None

    def check_prostate_mask_is_downloaded(self):
        fpath = os.path.join(self.directory, "prostate_mask.npy")
        return os.path.isfile(fpath)

    @property
    def bmode(self):
        if not os.path.isfile(os.path.join(self.directory, "bmode.npy")):
            return None
        return np.load(os.path.join(self.directory, "bmode.npy"))

    def download_bmode(self):
        iq = load_by_core_specifier(self.specifier)
        rf = DEFAULT_PREPROCESS_TRANSFORM(iq)

        bmode = to_bmode(rf)
        np.save(os.path.join(self.directory, "bmode.npy"), bmode)

    def download_prostate_mask(self):
        try:
            path = os.path.join(self.directory, "prostate_mask.npy")
            mask = load_prostate_mask(self.specifier, source="old_masks")
            np.save(path, mask)  # type:ignore
            return True
        except KeyError:
            return False

    @property
    def needle_mask(self):
        from exactvu.resources import needle_mask

        return needle_mask()

    @property
    def metadata(self):
        if self._metadata is None:
            from exactvu.resources import metadata

            _metadata = metadata()
            _metadata = _metadata.query(f"core_specifier == @self.specifier")
            self._metadata = dict(_metadata.iloc[0])

        return self._metadata

    def get_needle_mask_intersections(self, subpatch_size):
        fpath = os.path.join(
            self.directory,
            f"needle_mask_grid_subpatch_size_{subpatch_size[0]}-{subpatch_size[1]}.npy",
        )
        try:
            return np.load(fpath)
        except FileNotFoundError:
            intersections = self.compute_patch_intersections(
                self.needle_mask, subpatch_size
            )
            np.save(fpath, intersections)
            return intersections

    def get_prostate_mask_intersections(self, subpatch_size):
        fpath = os.path.join(
            self.directory,
            f"prostate_mask_grid_subpatch_size_{subpatch_size[0]}-{subpatch_size[1]}.npy",
        )
        try:
            return np.load(fpath)
        except FileNotFoundError:
            intersections = self.compute_patch_intersections(
                self.prostate_mask, subpatch_size
            )
            np.save(fpath, intersections)
            return intersections

    def download_and_preprocess_iq(
        self, iq_preprocessor_fn=DEFAULT_PREPROCESS_TRANSFORM
    ):
        iq = load_by_core_specifier(self.specifier)
        image = iq_preprocessor_fn(iq)
        self.image = image

    def get_grid_view(self, subpatch_size_mm=(1, 1)):

        assert self.image is not None, "Image not downloaded. "
        subpatch_size_pixels = patch_size_mm_to_pixels(
            self.image.shape, *subpatch_size_mm
        )

        if self.backend == PatchBackends.IN_MEMORY_IMAGE:

            grid = InMemoryImagePatchGrid(
                self.image,
                subpatch_size_pixels[0],
                subpatch_size_pixels[1],
            )

            return grid

        elif self.backend == PatchBackends.INDIVIDUAL_SAVED_SUBPATCHES:

            grid = SavedSubPatchGrid(
                os.path.join(self.directory, "image.npy"), *subpatch_size_pixels
            )
            return grid

        else:
            raise ValueError(f"backend {self.backend} not supported.")

    def compute_patch_intersections(self, mask, subpatch_size_mm):
        assert (img := self.image) is not None, "Image not loaded for core."
        return mask_intersections_for_grid(img.shape, mask, subpatch_size_mm)

    @overload
    def get_patch_view(
        self,
        patch_size=(5, 5),
        patch_strides=(1, 1),
        subpatch_size=(1, 1),
        needle_region_only=False,
        prostate_region_only=False,
        prostate_intersection_threshold=0.6,
        needle_intersection_threshold=0.6,
    ) -> PatchView:
        ...

    @overload
    def get_patch_view(self, config: PatchViewConfig) -> PatchView:
        ...

    def get_patch_view(self, *args, **kwargs):
        if len(args) == 1:
            config = args[0]
            assert isinstance(config, PatchViewConfig)
            return self._get_patch_view_impl(**asdict(config))
        else:
            return self._get_patch_view_impl(*args, **kwargs)

    def _get_patch_view_impl(
        self,
        patch_size=(5, 5),
        patch_strides=(1, 1),
        subpatch_size=(1, 1),
        needle_region_only=False,
        prostate_region_only=False,
        prostate_intersection_threshold=0.6,
        needle_intersection_threshold=0.6,
    ) -> PatchView:

        base_grid = self.get_grid_view(subpatch_size)

        h, w = base_grid.shape

        axial_startpos = [
            i for i in range(0, h, patch_strides[0]) if i + patch_size[0] <= h
        ]
        lateral_startpos = [
            i for i in range(0, w, patch_strides[1]) if i + patch_size[1] <= w
        ]

        patch_positions = []

        if needle_region_only:
            needle_intersections_grid = self.get_needle_mask_intersections(
                subpatch_size
            )
        else:
            needle_intersections_grid = None

        if prostate_region_only:
            if self.prostate_mask is None:
                raise ValueError(
                    f"""Requested grid view for prostate region only,
                    but prostate mask is unavailable for core {self.specifier}"""
                )
            prostate_intersections_grid = self.get_prostate_mask_intersections(
                subpatch_size
            )
        else:
            prostate_intersections_grid = None

        for i, j in product(axial_startpos, lateral_startpos):

            if needle_region_only:

                intersection = needle_intersections_grid[  # type:ignore
                    i : i + patch_size[0], j : j + patch_size[1]
                ]
                intersection = intersection.mean()

                if intersection < needle_intersection_threshold:
                    continue

            if prostate_region_only:

                intersection = prostate_intersections_grid[  # type:ignore
                    i : i + patch_size[0], j : j + patch_size[1]
                ]
                intersection = intersection.mean()

                if intersection < prostate_intersection_threshold:
                    continue

            patch_positions.append((i, i + patch_size[0], j, j + patch_size[0]))

        return PatchView(base_grid, patch_positions)

    def get_sliding_window_view(
        self, window_size=(5, 5), step_size=(1, 1), subpatch_size_mm=(1, 1)
    ):
        assert self.image is not None, "Download rf first"

        grid = self.get_grid_view(subpatch_size_mm)

        assert isinstance(
            grid, SubPatchAccessorMixin
        ), f"Backend {type(grid)} not supported for this operation."

        from .utils import sliding_window_grid

        out = {}

        grid_slices = sliding_window_grid(grid.shape, window_size, step_size)

        out["view"] = grid.view(grid_slices)
        out["positions"] = grid_slices
        rf_slices = grid_slices_to_rf_slices(
            grid_slices, self.image.shape, subpatch_size_mm
        )
        if self.needle_mask is not None:
            out["needle_region_intersections"] = mask_intersections_for_rf_slices(
                self.needle_mask, rf_slices, self.image.shape
            )

        if self.prostate_mask is not None:
            out["prostate_region_intersections"] = mask_intersections_for_rf_slices(
                self.needle_mask, rf_slices, self.image.shape
            )

        return out

    def filter_inside_needle_region(
        self, positions: Iterable, subpatch_size=(1, 1), threshold=0.65
    ):
        """
        Given the iterable of positions, returns a list of the positions that are inside the
        needle region for this core, according to the threshold specified. The patch will be considered
        inside if the fraction of its pixels lying inside the mask meets or exceeds the threshold.
        """

        filtered = []
        needle_intersections_grid = self.get_needle_mask_intersections(subpatch_size)

        for position in positions:

            x1, x2, y1, y2 = position

            intersection = needle_intersections_grid[  # type:ignore
                x1:x2, y1:y2
            ]
            intersection = intersection.mean()

            if intersection >= threshold:
                filtered.append(position)

        return filtered

    def filter_inside_prostate_region(
        self, positions, subpatch_size=(1, 1), threshold=0.9
    ):
        """
        Given the iterable of positions, returns a list of the positions that are inside the
        needle region for this core, according to the threshold specified. The patch will be considered
        inside if the fraction of its pixels lying inside the mask meets or exceeds the threshold.
        """
        filtered = []
        needle_intersections_grid = self.get_needle_mask_intersections(subpatch_size)

        for position in positions:

            x1, x2, y1, y2 = position

            intersection = needle_intersections_grid[  # type:ignore
                x1:x2, y1:y2
            ]
            intersection = intersection.mean()

            if intersection >= threshold:
                filtered.append(position)

        return filtered

    def clear_data(self):
        import shutil

        shutil.rmtree(self.directory)


def sample_core(
    dir=None,
) -> Core:
    """Returns a sample core object to speed up development"""
    from .splits import get_splits, filter_splits, HasProstateMaskFilter
    from .utils import data_dir

    train, val, test = filter_splits(
        get_splits("UVA600", True, 26), HasProstateMaskFilter()
    )
    return Core(train[0], dir)
