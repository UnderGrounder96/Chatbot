#!/usr/bin/env python3
import torch
import torch.nn as nn


class NeuralNet(nn.Module):
    """Deep learning magic"""

    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        self.output = self.l1(x)
        self.output = self.relu(self.output)
        self.output = self.l2(self.output)
        self.output = self.relu(self.output)
        
        # no activation and no softmax at the end
        return self.l3(self.output)