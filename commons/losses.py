from torch import Tensor, nn
import numpy as np
import torch.nn.functional as F
from typing import Tuple
from torch import Tensor

class ppiCrossEntropy(nn.Module):
    def __init__(self, weight=None) -> None:
        super(ppiCrossEntropy, self).__init__()
        self.weight = weight

    def forward(self, prediction: Tensor, target: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        crossentropy_loss = F.cross_entropy(prediction, target, weight=self.weight)
        loss = crossentropy_loss
        return loss, {'crossentropy_loss': crossentropy_loss}