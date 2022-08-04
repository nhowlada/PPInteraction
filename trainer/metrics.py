from typing import Union, List

import torch
from torch.nn import functional as F
import torch.nn as nn

from sklearn import metrics

# https://towardsdatascience.com/the-5-classification-evaluation-metrics-you-must-know-aa97784ff226
class Bal_Accuracy(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.balanced_accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return score

class Accuracy(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.accuracy_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return score
class MCC(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.matthews_corrcoef(targets.detach().cpu().numpy(), preds.detach().cpu().numpy())
        return score
class F1_Score(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.f1_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy(), zero_division=1)
        return score
class AUC(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss
class Precision(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.precision_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy(), zero_division=1)
        return score
class Recall(nn.Module):
    def __init__(self, ):
        super().__init__()
    def forward(self, prediction, targets):
        preds = torch.max(prediction[..., :2], dim=1)[1]
        score = metrics.recall_score(targets.detach().cpu().numpy(), preds.detach().cpu().numpy(), zero_division=1)
        return score

class MAE(nn.Module):
    def __init__(self, ):
        super().__init__()

    def forward(self, preds, targets):
        loss = F.l1_loss(preds, targets)
        return loss