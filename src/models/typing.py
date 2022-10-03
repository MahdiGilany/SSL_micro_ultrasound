from typing import Protocol, runtime_checkable
import torch


@runtime_checkable
class FeatureExtractionProtocol(Protocol):

    features_dim: int

    def get_features(self, X: torch.Tensor) -> torch.Tensor:
        ...
