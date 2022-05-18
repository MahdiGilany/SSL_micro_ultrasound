from itertools import product

import numpy as np
from ismember import ismember

from torch.utils.data import Dataset
from pytorch_lightning.core.mixins import HyperparametersMixin

from .data_utils import *


class ExactDataset(Dataset, HyperparametersMixin):
    """Characterizes a dataset for PyTorch."""

    def __init__(
            self,
            state: str,
            dataset_hyp: dict,
            extended_metadata: dict
    ):
        super().__init__()

        """
        # save all input parameters in hparams including:
        #   dataset_hyp.inv_cutoff
        #   dataset_hyp.patch_sz
        #   dataset_hyp.jump_sz
        #   dataset_hyp.aug_list
        #   dataset_hyp.aug_prob
        #   dataset_hyp.SSL

        #   # extended_metadata.meta_data
        #   # extended_metadata.patch_centers_sl
        #   # extended_metadata.data_roots
        """
        self.save_hyperparameters(logger=False)

        # patch centers of the whole RF image. List [axials, laterals]
        patch_centers_RFimg = extended_metadata.meta_data['patch_centers1x1']
        self.patch_centers_RFimg: list = [np.unique(patch_centers_RFimg[:, 0]), np.unique(patch_centers_RFimg[:, 1])]

        # finding len and name of patch centers in out dataset based on patch centers inside
        # needle for SL and inside the whole image in SSL (former not implemented yet)
        self.len_ds, self.used_patch_names, self.core_lengths = self.find_len_and_centers()

        # RF image index number for each of used patches (used_patch_names). These indexes can be used for
        # accessing information in meta data corresponding to train/test/val patches
        self.ind_RFimg = [self.patchind_to_RFimgind(i) for i, n in enumerate(self.used_patch_names)]

        # labels
        self.core_labels = extended_metadata.meta_data[f'label_{self.hparams.state}']
        self.labels = [self.core_labels[ind] for i, ind in enumerate(self.ind_RFimg)]
        # self.labels = to_categorical(self.label)

        # augmentation
        self.transforms = aug_transforms(state, dataset_hyp.aug_list, p=dataset_hyp.aug_prob)\
            if self.hparams.dataset_hyp.SSL else None

    def find_len_and_centers(self):
        """finds the names of central patches and len of that which correspond to len of data since
        for each central patch we would have one bigger patch."""
        state = self.hparams.state
        meta_data = self.hparams.extended_metadata.meta_data

        data_names = meta_data[f'data_{state}']

        # appending name of all central patches (1x1 patches) inside needle and prostate from RF images in data_{state}
        all_patchnames_sl_pr = []
        for i, patch1x1_name_pr in enumerate(data_names):
            # """looping over all RF images and multiplying mask of prostate & needle and selecting
            # centers of 5x5 patches.

            # first get all centers of patches inside prostate for i'th RF image
            all_1x1patchcenters_pr = [np.array(name.split('_')[-3:-1]).astype(int) for name in patch1x1_name_pr]
            all_1x1patchcenters_pr = np.stack(all_1x1patchcenters_pr)

            # first get all names of patches inside prostate for i'th RF image
            ## todo: the format of files are set here. Needs to change.
            all_1x1patchnames_pr = [np.array('_'.join(name.split('_')[:-1]) + '_frm1.pkl') for name in patch1x1_name_pr]
            all_1x1patchnames_pr = np.stack(all_1x1patchnames_pr)

            # retaining only patches inside needle (inside extended_metadata.patch_centers_sl)
            ind, _ = ismember(all_1x1patchcenters_pr, self.hparams.extended_metadata.patch_centers_sl, "rows")
            #   #validate if bigger patches are inside prostate as well, in addition to central 1x1 patches.
            ind = self.validate_mask_ind(all_1x1patchcenters_pr, ind)
            all_patchnames_sl_pr.append(all_1x1patchnames_pr[ind])

        # corelen for all cores in supervised learning setting (number of patches in each core)
        all_corelen_sl = [core.shape[0] for core in all_patchnames_sl_pr]

        return np.sum(all_corelen_sl), np.concatenate(all_patchnames_sl_pr), all_corelen_sl

    def validate_mask_ind(self, all_1x1patchcenters_pr, ind):
        """checking head and tail big patches to see if they are completely located in prostate
        region corner patches of head and tail patches are only checked."""

        # patch size for finding the corner of bigger patch
        patch_sz = self.hparams.dataset_hyp.patch_sz

        # axial and lateral numbers corresponding to patch centers in the whole RF image (used to find corner patches)
        patch_centers_RFimg = self.patch_centers_RFimg
        axl_center_numbers = patch_centers_RFimg[0]
        lat_center_numbers = patch_centers_RFimg[1]

        # centers of central patches inside both needle and prostate
        all_patchcenters_sl_pr = all_1x1patchcenters_pr[ind]

        # patches to be checked
        chk_centers = range(len(all_patchcenters_sl_pr))
        # chk_centers = [0, 1, 2, -1, -2, -3]
        # if there is no or less than 3 patches inside both prostate and needle masks
        # if len(all_patchcenters_sl_pr) <= 5:
        #     chk_centers = list(range(len(all_patchcenters_sl_pr)))

        # span of axial and lateral indices
        hlf_patch_sz = np.floor((patch_sz-1)/2.).astype(int)
        hlf_patch_sz_rest = np.ceil((patch_sz-1)/2.).astype(int)

        copy_ind = np.copy(ind)
        for center in chk_centers:
            # checking head and tail big patches to see if they are completely located in prostate
            # region.

            # indexes of possible axial and lateral numbers
            cent_axl_ind, = np.where(axl_center_numbers == all_patchcenters_sl_pr[center, 0])
            cent_lat_ind, = np.where(lat_center_numbers == all_patchcenters_sl_pr[center, 1])

            # axial and lateral numbers of left bottom corner
            ## todo: this redundancy should be modified
            corner_axl = axl_center_numbers[cent_axl_ind+hlf_patch_sz_rest][0]
            corner_lat = lat_center_numbers[cent_lat_ind-hlf_patch_sz][0]
            member_bool1 = ismember(np.array([[corner_axl, corner_lat]]), all_1x1patchcenters_pr, "rows")[0]

            # axial and lateral numbers of top right corner
            corner_axl = axl_center_numbers[cent_axl_ind-hlf_patch_sz][0]
            corner_lat = lat_center_numbers[cent_lat_ind+hlf_patch_sz_rest][0]
            member_bool2 = ismember(np.array([[corner_axl, corner_lat]]), all_1x1patchcenters_pr, "rows")[0]

            # axial and lateral numbers of bottom right corner
            corner_axl = axl_center_numbers[cent_axl_ind+hlf_patch_sz_rest][0]
            corner_lat = lat_center_numbers[cent_lat_ind+hlf_patch_sz_rest][0]
            member_bool3 = ismember(np.array([[corner_axl, corner_lat]]), all_1x1patchcenters_pr, "rows")[0]

            # axial and lateral numbers of top left corner
            corner_axl = axl_center_numbers[cent_axl_ind-hlf_patch_sz][0]
            corner_lat = lat_center_numbers[cent_lat_ind-hlf_patch_sz][0]
            member_bool4 = ismember(np.array([[corner_axl, corner_lat]]), all_1x1patchcenters_pr, "rows")[0]

            # if any of the corners are outside prostate, change the index of that patch to be False rather than True
            if (not member_bool1) or (not member_bool2) or (not member_bool3) or (not member_bool4):
                ind_of_ind, = np.where(ind == True)
                copy_ind[ind_of_ind[center]] = False

        return copy_ind

    def __len__(self):
        return self.len_ds

    def __getitem__(self, index):
        x_patch, y_target = self.get_img_target(index)

        # apply transformations to x_patch
        x_patch1 = apply_transforms(x_patch, self.transforms) if self.transforms is not None else x_patch

        if self.transforms is not None:
            x_patch2 = apply_transforms(x_patch, self.transforms)
            return [x_patch1, x_patch2], y_target

        return x_patch1, y_target

    def get_img_target(self, index):
        # name of central patch that should be used for creating the whole patch
        patch_name = self.used_patch_names[index]

        # finding patch root
        patch_parts = patch_name.split('_')
        patch_root = '_'.join(patch_parts[:-3])
        centers_centralpatch = np.array(patch_parts[-3:-1]).astype(int)

        # all roots of data
        data_roots = self.hparams.extended_metadata.data_roots

        # getting the complete patch root
        complete_fileroot = [f for f in data_roots if patch_root in f]
        complete_fileroot = os.path.join(complete_fileroot[0], patch_root)

        # loading and creating the patch
        x_patch = self.load_create_patch(complete_root=complete_fileroot, centers_centralpatch=centers_centralpatch,
                                         suffix=patch_parts[-1])

        # assigning core label to patch label
        y_label = self.labels[index]
        return x_patch, y_label

    def load_create_patch(self, complete_root, centers_centralpatch, suffix):
        # patch size for creating bigger patch
        patch_sz = self.hparams.dataset_hyp.patch_sz
        patch_centers_RFimg = self.patch_centers_RFimg


        # finding index of cental patch in the RF img
        axl_center_numbers = patch_centers_RFimg[0]
        lat_center_numbers = patch_centers_RFimg[1]
        cent_axl_ind, = np.where(axl_center_numbers == centers_centralpatch[0])
        cent_lat_ind, = np.where(lat_center_numbers == centers_centralpatch[1])

        # span of axial and lateral indices
        hlf_patch_sz = np.floor((patch_sz-1)/2.).astype(int)
        hlf_patch_sz_rest = np.ceil((patch_sz-1)/2.).astype(int)
        cent_axl_range_ind = np.arange(cent_axl_ind[0]-hlf_patch_sz, cent_axl_ind[0]+hlf_patch_sz_rest+1)
        cent_lat_range_ind = np.arange(cent_lat_ind[0]-hlf_patch_sz, cent_lat_ind[0]+hlf_patch_sz_rest+1)

        all_1x1patches = []
        for axl_ind, lat_ind in product(cent_axl_range_ind,cent_lat_range_ind):
            patch_name = complete_root + f'_{axl_center_numbers[axl_ind]}_{lat_center_numbers[lat_ind]}_' + suffix
            all_1x1patches.append(load_pickle(patch_name)) #['data'])

        # stitching patches together
        all_1x1patches = np.stack(all_1x1patches)
        all_1x1patches = rearrange(all_1x1patches, '(b1 b2) h w -> (b1 h) (b2 w)', b1=patch_sz) #25x360x11 to 5x360 x 5x11
        return resize_norm(all_1x1patches)[np.newaxis, ...]

    def patchind_to_RFimgind(self, index):
        # loading all corelen
        corelen_sl = self.core_lengths

        # cumulative sum of corelen to find where index is located
        cumsum_corelen = np.cumsum(corelen_sl)
        ind, = np.where(index <= cumsum_corelen)
        RFimg_ind = ind[0]

        return RFimg_ind
