import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, obs_shape, out_features, channels, kernels, strides, paddings):
        super(CNN, self).__init__()
        layers = []
        for i in range(len(channels) - 1):
            layers.append(nn.Conv2d(channels[i], channels[i+1], kernels[i], strides[i], paddings[i]))
            layers.append(nn.ReLU())
        layers.append(nn.Flatten())
        self.conv_layers = nn.Sequential(*layers)

        with torch.no_grad():
            n_flatten = self.conv_layers(torch.zeros(1, *obs_shape)).shape[1]

        self.fc = nn.Sequential(nn.Linear(n_flatten, out_features), nn.ReLU())
    
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(0)
        x = self.conv_layers(x)
        x = self.fc(x)
        return x

