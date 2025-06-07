import torch
from torch import nn
from torchinfo import summary
import torch.nn.functional as F

import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.relu = nn.ReLU()
        self.c1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.s2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c3 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.s4 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.c5 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.c6 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.c7 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.s8 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.f1 = nn.Linear(6 * 6 * 256, out_features=4096)
        self.f2 = nn.Linear(in_features=4096, out_features=4096)
        self.f3 = nn.Linear(in_features=4096, out_features=10)

    def forward(self, x):
        x = self.relu(self.c1(x))
        x = self.s2(x)
        x = self.relu(self.c3(x))
        x = self.s4(x)
        x = self.relu(self.c5(x))
        x = self.relu(self.c6(x))
        x = self.relu(self.c7(x))
        x = self.s8(x)

        x = self.flatten(x)
        x = self.relu(self.f1(x))
        x = F.dropout(x, p = 0.5)
        x = self.relu(self.f2(x))
        x = F.dropout(x, p = 0.5)
        x = self.f3(x)
        return x
    
    if __name__ == '__main__':
        device = torch.device("mps")

        model = Alexnet().to(device)
        print(summary())
