import torch.nn as nn
from .bert import SBERT

class SBERTPrediction(nn.Module):
    """
    Pre-training task: predicting contaminated observations given an entire annual satellite time series
    """

    def __init__(self, sbert: SBERT, num_features):
        super().__init__()
        self.sbert = sbert
        self.mask_prediction = MaskedTimeSeriesModel(self.sbert.hidden, num_features)

    def forward(self, x, doy, mask):
        x = self.sbert(x, doy, mask)
        return self.mask_prediction(x)


class MaskedTimeSeriesModel(nn.Module):

    def __init__(self, hidden, num_features):
        super().__init__()
        self.hidden = hidden
        self.num_features = num_features
        
        self.linear1 = nn.Linear(hidden, hidden)
        self.linear2 = nn.Linear(hidden, hidden)
        self.linear3 = nn.Linear(hidden, num_features)
        
        self.layer_norm1 = nn.LayerNorm(hidden)
        self.layer_norm2 = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(p=0.1)
        
        self.attention = nn.MultiheadAttention(embed_dim=hidden, num_heads=8)

    def forward(self, x):
        x, _ = self.attention(x, x, x)
        
        x = self.layer_norm1(x)
        x = self.dropout(nn.functional.gelu(self.linear1(x)))
        
        x = self.layer_norm2(x)
        x = self.dropout(nn.functional.gelu(self.linear2(x)))
        
        x = self.linear3(x)
        
        return x