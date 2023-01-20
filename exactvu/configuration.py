"""
Registers several config schema to use with hydra 
(see )
""" 


from .data.core import PatchViewConfig
from .data.transforms import (
    TransformConfig,
    NormConfig,
    UltrasoundAugsConfig,
    TensorAugsConfig,
)
from .data.splits import SplitsConfig
from .data.datamodule import ExactPatchDMConfig


def register_configs():
    from hydra.core.config_store import ConfigStore
    store = ConfigStore.instance()

    store.store('transform_config', TransformConfig, 'exactvu')
    store.store('patch_view_config', PatchViewConfig, 'exactvu')
    store.store('splits_config', SplitsConfig, 'exactvu')