import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 2)
        self.activate = nn.Tanh()

    def forward(self, x):
        x1 = self.fc1(x)
        x2 = self.activate(x1)
        x3 = self.fc2(x2)
        x4 = self.activate(x3)
        x5 = self.fc3(x4)
        return x5


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc1 = nn.Linear(2, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        self.activate = nn.Tanh()

    def forward(self, x):
        x1 = x.type(torch.float32)
        x2 = self.fc1(x1)
        x3 = self.activate(x2)
        x4 = self.fc2(x3)
        x5 = self.activate(x4)
        x6 = self.fc3(x5)
        return torch.sigmoid(x6), x5
