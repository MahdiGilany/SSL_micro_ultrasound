
from omegaconf import OmegaConf


def checkpoint_reader(fpath, idx): 
    with open(fpath, 'r') as f: 
        return list(f)[idx]


def register_resolvers(): 
    OmegaConf.register_new_resolver('ckpt', checkpoint_reader)