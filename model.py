import torch
import torch.nn as nn

def he_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)

class CustomSigmoid(nn.Module):
    def __init__(self, scalar=1.83804/400):
        super(CustomSigmoid, self).__init__()
        self.scalar = scalar

    def forward(self, x):
        return 1 / (1 + torch.exp(-x * self.scalar))

class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256*2, 1)  
        self.custom_sigmoid = CustomSigmoid()  

        self.apply(he_init)

    def forward(self, x1, x2):
        x1 = torch.relu(self.fc1(x1))
        x2 = torch.relu(self.fc1(x2))

        merged_input = torch.cat((x1, x2), dim=1)

        x = self.custom_sigmoid(self.fc2(merged_input))
        return x
    