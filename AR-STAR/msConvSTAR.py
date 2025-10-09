# This is the multi-stage convSTAR
import torch
import torch.nn as nn
import torch.nn.functional as F

from convSTAR import ConvSTAR

class ConvRNN(nn.Module):
    """A ConvRNN model for pixel-wise binary classification."""
    
    def __init__(self, height=24, width=24, input_dim=10, hidden_dim=64, kernel_size=(3,3), n_layers=6):
        super(ConvRNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        # ConvSTAR-based Recurrent Layer
        self.rnn = ConvSTAR(input_size=input_dim,
                            hidden_sizes=hidden_dim,
                            kernel_sizes=kernel_size[0],
                            n_layers=n_layers)

        # Final Conv Layer: Output 1 channel (Binary Classification)
        self.final = nn.Conv2d(hidden_dim, 1, kernel_size=(3, 3), padding=1)

    def forward(self, x, hiddenS=None):
        """
        Input Shape: (batch_size, time_steps, channels, height, width) -> (B, 28, 10, 24, 24)
        """
        # (B, T, C, H, W) → (B, C, T, H, W)
        x = x.permute(0, 2, 1, 3, 4)

        b, c, t, h, w = x.shape

        # Initialize hidden state if not provided
        if hiddenS is None:
            hiddenS = [torch.zeros((b, self.hidden_dim, h, w))] * self.n_layers
            
        if torch.cuda.is_available():
            hiddenS = [h.cuda() for h in hiddenS]

        # Process through the ConvRNN
        for iter in range(t):
            hiddenS = self.rnn(x[:, :, iter, :, :], hiddenS)

        # Get the last hidden state and apply final classification layer
        last = hiddenS[-1]  # Take the last layer's output
        # print(last)
        last = self.final(last)  # (B, 1, H, W)

        return last  # Output is probability (0-1) for each pixel