import torch
import torch.nn as nn
from .bert import SBERT

class SBERTClassification(nn.Module):
    """
    Downstream task: Satellite Time Series Classification
    """

    def __init__(self, sbert: SBERT, num_classes):
        super().__init__()
        self.sbert = sbert
        self.classification = MulticlassClassification(self.sbert.hidden, num_classes)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.classification(x, mask)


class MulticlassClassification(nn.Module):

    def __init__(self, hidden, num_classes):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.3)
        self.linear1 = nn.Linear(hidden, hidden // 2)
        self.layer_norm = nn.LayerNorm(hidden // 2)
        self.linear2 = nn.Linear(hidden // 2, num_classes)

    def forward(self, x, mask):
        # Apply pooling
        x = self.pooling(x.permute(0, 2, 1)).squeeze(-1)
        
        # Apply dropout
        x = self.dropout(x)
        
        # First linear layer with layer normalization and dropout
        x = nn.functional.gelu(self.linear1(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        
        # Final linear layer for classification
        x = self.linear2(x)
        
        return x