from typing import Tuple, Union

import torch
import numpy as np

from commons.utils import TARGET, SOLUBILITY


class ToTensor():


    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, np.ndarray, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding1, embedding2, target = sample
        embedding1 = torch.tensor(embedding1).float()
        embedding2 = torch.tensor(embedding2).float()
        target = torch.tensor(target).long()
        return embedding1, embedding2, target


class AvgMaxPool():
    """
    """

    def __init__(self, dim: int = -2):
        """
        """
        self.dim = dim

    def __call__(self, sample: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        """
        embedding, localization, solubility = sample
        avg_pool = torch.mean(embedding, dim=self.dim)
        max_pool, _ = torch.max(embedding, dim=self.dim)
        embedding = torch.cat([avg_pool, max_pool], dim=-1)
        return embedding, localization, solubility

class LabelOneHot():
    """
    """

    def __init__(self):
        pass

    def __call__(self, sample: Tuple[np.ndarray, str, str]) -> Tuple[np.ndarray, np.ndarray, int]:
        """
        """
        embedding, localization, solubility = sample
        localization = TARGET.index(localization)
        one_hot_localization = np.zeros(len(TARGET))
        one_hot_localization[localization] = 1
        solubility = SOLUBILITY.index(solubility)
        return embedding, one_hot_localization, solubility
