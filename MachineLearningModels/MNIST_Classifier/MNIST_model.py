import os
import torch
from torch import nn

device = (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(16, 8, 3, stride=1, padding=1)

        # LeakyReLU activation
        self.leaky_relu = nn.LeakyReLU(0.1)

        # Max pooling
        self.max_pool = nn.MaxPool2d(3, 3)

        # Dropout
        self.dropout = nn.Dropout(0.25)

        # Flatten layer
        self.flatten = nn.Flatten()

        # Fully connected layer
        self.fc = nn.Linear(8 * 9 * 9, 10)  # in_features depends on the output size of the last conv layer

    def forward(self, x):
        x = self.conv1(x)  # (28, 28, 8)
        x = self.leaky_relu(x)
        x = self.conv2(x)  # (28, 28, 16)
        x = self.leaky_relu(x)
        x = self.max_pool(x)  # (9, 9, 16)
        x = self.dropout(x)
        x = self.conv3(x)  # (9, 9, 8)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.flatten(x)  # (8, 81)
        x = self.fc(x)
        return x



