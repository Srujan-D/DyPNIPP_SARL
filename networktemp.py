import torch
import torch.nn as nn
import math
import numpy as np
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp.autocast_mode import autocast


class PredictNextBelief(nn.Module):
    def __init__(self, device="cuda"):
        super(PredictNextBelief, self).__init__()
        self.device = device
        # self.conv encoder --> what features of GP to use? predicted mean, uncertainty, 
                            # --> do we want to encode history (just regress GP over time) explicitly?
        # self.lstm layer
        # self.MLP layer

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(8, 4, kernel_size=(3, 3)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

    def forward(self, x):
        x = self.conv_encoder(x)
        return x
    

if __name__ == "__main__":
    model = PredictNextBelief()
    dummy_ip = torch.randn(1, 1, 30, 30)
    out = model(dummy_ip)
    print(out.shape)

