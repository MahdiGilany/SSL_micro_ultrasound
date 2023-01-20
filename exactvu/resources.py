from copyreg import pickle
import numpy as np
import pkg_resources
import pandas as pd


def needle_mask():
    mask_fname = pkg_resources.resource_filename(__name__, "resources/needle_mask.npy")
    needle_mask = np.load(mask_fname)
    return needle_mask


def metadata():
    metadata_fname = pkg_resources.resource_filename(__name__, "resources/metadata.csv")
    metadata = pd.read_csv(metadata_fname, index_col=[0])
    metadata["patient_id"] = metadata["patient_id"].astype("int")
    return metadata


def miccai_splits():
    metadata_fname = pkg_resources.resource_filename(
        __name__, "resources/miccai_2022_patient_groups.csv"
    )
    miccai_splits = pd.read_csv(metadata_fname)
    return miccai_splits


def patient_test_sets():
    fname = pkg_resources.resource_filename(
        __name__, "resources/patients_test_set_by_center.pkl"
    )

    with open(fname, "rb") as f:
        import pickle

        out = pickle.load(f)

    return out


def crceo_428_splits(): 
    metadata_fname = pkg_resources.resource_filename(
        __name__, "resources/crceo_428.csv"
    )
    splits = pd.read_csv(metadata_fname, index_col=0)
    return splits