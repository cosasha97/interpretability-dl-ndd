import numpy as np
from torchsummary import summary
from math import floor

# torch
import torch.nn as nn


class Net(nn.Module):
    def __init__(self, sample, convolutions):
        super().__init__()
        self.image_size = sample['image'].shape
        self.features = nn.Sequential()
        input_size = self.image_size[0]
        h_w = self.image_size[1:]

        # default parameters
        kernel_size = 3

        for index, nb_conv in enumerate(convolutions):
            self.features.add_module('conv' + str(index), nn.Conv2d(input_size, nb_conv, kernel_size))
            h_w = self.conv_output_shape(h_w, 3)
            self.features.add_module('relu' + str(index), nn.ReLU())
            self.features.add_module('bnn' + str(index), nn.BatchNorm2d(nb_conv))
            self.features.add_module('pool' + str(index), nn.MaxPool2d(2, 2))
            h_w = self.conv_output_shape(h_w, 2, stride=2)
            input_size = nb_conv

        self.features_output_size = np.prod(h_w) * nb_conv
        self.dense_size_1 = 32
        self.dense_size_2 = 16

        # classifyer
        self.branch1 = nn.Sequential(
            nn.Linear(self.features_output_size, self.dense_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_1),
            nn.Linear(self.dense_size_1, self.dense_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_2),
            nn.Linear(self.dense_size_2, 1),
            nn.Sigmoid()
        )
        # volumes
        n_volumes = np.prod(sample['volumes'].shape)
        self.branch2 = nn.Sequential(
            nn.Linear(self.features_output_size, 4 * n_volumes),
            nn.ReLU(),
            nn.BatchNorm1d(4 * n_volumes),
            nn.Linear(4 * n_volumes, 2 * n_volumes),
            nn.ReLU(),
            nn.BatchNorm1d(2 * n_volumes),
            nn.Linear(2 * n_volumes, n_volumes)
        )
        # age
        self.branch3 = nn.Sequential(
            nn.Linear(self.features_output_size, self.dense_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_1),
            nn.Linear(self.dense_size_1, self.dense_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_2),
            nn.Linear(self.dense_size_2, 1)
        )
        # sex
        self.branch4 = nn.Sequential(
            nn.Linear(self.features_output_size, self.dense_size_1),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_1),
            nn.Linear(self.dense_size_1, self.dense_size_2),
            nn.ReLU(),
            nn.BatchNorm1d(self.dense_size_2),
            nn.Linear(self.dense_size_2, 1),
            nn.Sigmoid()
        )

    @staticmethod
    def conv_output_shape(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
        if type(kernel_size) is not tuple:
            kernel_size = (kernel_size, kernel_size)
        h = floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
        w = floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
        return h, w

    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.features_output_size)
        c = self.branch1(x)
        v = self.branch2(x)
        age = self.branch3(x)
        sex = self.branch4(x)
        return c, v, age, sex

    def summary(self):
        """
        Print a summary of the model.
        """
        summary(self, input_size=self.image_size)
