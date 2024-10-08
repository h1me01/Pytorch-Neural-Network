import torch
import torch.nn as nn

class MPE(nn.Module):
    def __init__(self, power=2):
        super(MPE, self).__init__()
        self.power = power

    def forward(self, outputs, targets):
        diff = outputs - targets
        abs_diff = torch.abs(diff)
        return torch.mean(torch.pow(abs_diff, self.power))
    