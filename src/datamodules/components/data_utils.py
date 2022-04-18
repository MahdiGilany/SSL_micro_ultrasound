import os
import pickle

import numpy as np
import pandas as pd

from PIL import Image
from einops import rearrange
from mat73 import loadmat as loadmat73
from scipy.io import matlab
from sklearn.model_selection import GroupShuffleSplit

from torchvision.transforms import transforms


def load_matlab(filename):
    with open(filename, 'rb') as fp:
        return matlab.loadmat(fp, struct_as_record=False, squeeze_me=True)


def load_matlab73(filename):
    return loadmat73(filename)


def load_pickle(filename):
    with open(filename, 'rb') as fp:
        return pickle.load(fp)


def to_categorical(target):
    n_classes = np.max(target) + 1
    y_c = np.zeros((len(target), np.int(n_classes)))
    for i in range(len(target)):
        y_c[i, np.int(target[i])] = 1
    return y_c.astype(int)


def merge_split_train_val(meta_data, random_state=26, train_val_split=0.25):
    """

      :param meta_data:
      :param random_state:
      :param train_val_split:
      :return:
      """

    # merge train-val then randomize patient ID to split train-val again
    gs = list(meta_data["GS_train"]) + list(meta_data['GS_val'])
    pid = list(meta_data["PatientId_train"]) + list(meta_data['PatientId_val'])
    inv = list(meta_data["inv_train"]) + list(meta_data['inv_val'])

    df1 = pd.DataFrame({'pid': pid, 'gs': gs, 'inv': inv})
    df1 = df1.assign(condition=df1.gs.replace({'Benign': 'Benign', 'G7': 'Cancer', 'G8': 'Cancer',
                                               'G9': 'Cancer', 'G10': 'Cancer'}))

    train_inds, test_inds = next(GroupShuffleSplit(test_size=train_val_split, n_splits=2,
                                                   random_state=random_state).split(df1, groups=df1['pid']))
    df1 = df1.assign(group='train')
    df1.loc[test_inds, 'group'] = 'val'
    df1 = df1.sort_values(by='pid')

    pid_tv = {
        'train': df1[df1.group == 'train'].pid.unique(),
        'val': df1[df1.group == 'val'].pid.unique(),
    }

    # Merge train - val
    keys = [f[:-4] for f in meta_data.keys() if 'val' in f]
    merge = {}
    for k in keys:
        if isinstance(meta_data[f'{k}_train'], list):
            merge[k] = meta_data[f'{k}_train'] + meta_data[f'{k}_val']
        else:
            merge[k] = np.concatenate([meta_data[f'{k}_train'], meta_data[f'{k}_val']], axis=0)

    # Initialize the new meta_data
    target = {}
    for set_name in ['train', 'val']:
        for k in keys:
            target[f'{k}_{set_name}'] = []

    # Re-split data into two sets based on randomized patient ID
    for i, pid in enumerate(merge['PatientId']):
        for set_name in ['train', 'val']:
            if pid in pid_tv[set_name]:
                for k in keys:
                    k_target = f'{k}_{set_name}'
                    target[k_target].append(merge[k][i])

    # Assign to original input data after finishing creating a new one
    for set_name in ['train', 'val']:
        for k in keys:
            k_target = f'{k}_{set_name}'
            meta_data[k_target] = target[k_target]
            if isinstance(merge[k], np.ndarray):
                meta_data[k_target] = np.array(target[k_target]).astype(np.ndarray)

    return meta_data


def remove_empty_lowinv_data(meta_data, dataset_hyp=None):
    """
    :param meta_data:
    :return:
    """

    inv_cutoff = dataset_hyp.inv_cutoff

    for set_name in ['val', 'test', 'train']:
        data = meta_data[f"data_{set_name}"]
        inv = meta_data[f"inv_{set_name}"]

        # indexes to include: not empty data and with inv bigger that cutoff
        include_idx = [i for i, d in enumerate(data)
                       if (isinstance(d, np.ndarray) and (inv[i] >= inv_cutoff or inv[i] == 0.))]

        for k in meta_data.keys():
            if set_name in k:
                if isinstance(meta_data[k], list):
                    meta_data[k] = [_ for i, _ in enumerate(meta_data[k]) if i in include_idx]
                else:
                    meta_data[k] = meta_data[k][include_idx]
    return meta_data


def estimate_patchcenter(meta_data, dataset_hyp=None):
    """finds patch centers considering that only 1mmx1mm patches are accessible.

    Inspired from Amoon's matlab code
    """
    patch_sz = dataset_hyp.patch_sz
    jump_sz = dataset_hyp.jump_sz

    centers = meta_data['patch_centers1x1']
    needle_mask = meta_data['needle_mask']

    # changing dimensionality of patch centers
    centers = rearrange(centers, '(ax c) lat -> ax c lat', ax=28)

    # finding if center of a patch is inside needle mask
    patch_mask = needle_mask[centers[:, 0, 0][:, None], centers[:, :, 1]]

    # finding lateral index of central patch
    col_select = np.sum(patch_mask, axis=0)
    ind_lat = np.argwhere(col_select > 0)
    st_lat = ind_lat[0].item() + np.floor((patch_sz - 1) / 2.).astype(int)
    end_lat = ind_lat[-1].item() - np.ceil((patch_sz - 1) / 2.).astype(int)

    patch_centers_sl = []
    patch_center_ind = []
    for i in range(st_lat, end_lat + 1, jump_sz):
        # finding axial index of central patch
        ind_ax = np.argwhere(patch_mask[:, i] > 0)
        j = np.floor(ind_ax.mean()).astype(int)

        # finding actual patch center from indices and saving it
        patch_center_ind.append(np.array([[j, i]]))
        cent = np.array([[centers[j, 0, 0], centers[0, i, 1]]])
        patch_centers_sl.append(cent)

    patch_centers_sl = np.concatenate(patch_centers_sl)
    patch_center_ind = np.concatenate(patch_center_ind)
    return patch_centers_sl


def get_data_roots(data_dir):
    """finds roots of all data."""
    # list_all_files = []
    list_all_roots = []
    for root, dir, files in os.walk(data_dir):
        # file_dir = [root+'\\'+ f for i, f in enumerate(files)]
        # list_all_files.append(file_dir)
        list_all_roots.append(root)
    return list_all_roots


def resize_norm(patch):
    # interpolation and normalization of patch
    patch_resized = np.array(Image.fromarray(patch).resize((256, 256)))
    patch = (patch_resized - np.mean(patch_resized, keepdims=True)) / np.std(patch_resized, keepdims=True)

    # truncating patches to bring them to [0,1]
    patch[patch >= 4] = 4.
    patch[patch <= -4] = -4.
    return (patch - patch.min()) / (patch.max() - patch.min())
    # return (patch_resized - 0.24708273)/848.8191
    # return patch_resized


def aug_transforms(state, aug_list, p=.5):
    if state != 'train' or len(aug_list) == 0:
        return None

    aug_transforms = [transforms.ToTensor()]
    for i, aug in enumerate(aug_list):
        if aug == 'RandomInvert':
            aug_transforms.append(transforms.RandomInvert(p))
        elif aug == 'RandomVerticalFlip':
            aug_transforms.append(transforms.RandomVerticalFlip(p))
        elif aug == 'RandomHorizontalFlip':
            aug_transforms.append(transforms.RandomHorizontalFlip(p))
        elif aug == 'RandomAffine':
            aug_transforms.append(transforms.RandomAffine(degrees=(0, 0), translate=(0.3, 0.3), fill=0.5))
        elif aug == 'RandomEqualize':
            aug_transforms.append(transforms.RandomEqualize(p))
        elif aug == 'RandomErasing':
            aug_transforms.append(transforms.RandomErasing(p=p, scale=(0.01, 0.05), ratio=(0.3, 3.3), value=.5))

        return transforms.Compose(aug_transforms)


def apply_transforms(patch, transforms):
    patch = rearrange(patch, 'c h w -> h w c')
    #
    # # truncating patches to bring them to [0,1]
    # patch[patch >= 4] = 4.
    # patch[patch <= -4] = -4.
    # patch = (patch - patch.min()) / (patch.max() - patch.min())

    patch = transforms(patch)
    return patch
