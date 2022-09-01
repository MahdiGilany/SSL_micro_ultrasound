from ..evaluation_base import EvaluationBase, SharedStepOutput
import torch
from torch.nn import functional as F
from ..components.backbones import create_model


class SupervisedModel(EvaluationBase):
    def __init__(
        self,
        backbone: str,
        batch_size: int,
        epochs: int = 100,
        learning_rate: float = 0.1,
        **kwargs,
    ):
        super().__init__(
            batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, **kwargs
        )
        self.save_hyperparameters()
        self.backbone_name = backbone
        self.backbone = create_model(backbone)

    def shared_step(self, batch) -> SharedStepOutput:
        X, y, metadata = batch

        logits = self.backbone(X)

        loss = F.cross_entropy(logits, y)

        return SharedStepOutput(logits, y, loss, [metadata])

    def get_learnable_parameters(self):
        return self.backbone.parameters()
