import torch
from torch import nn
import numpy as np
import config as c


class FullConnectedNetwork(nn.Module):
    '''Convolutional Q Network'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Linear(c.INPUT, 64),
                                            nn.ReLU(),
                                            nn.Linear(64,64),
                                            nn.ReLU(),
                                            nn.Linear(64, c.OUTPUT))
        self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        self.y_output = self.stacked_layers(x_input)
        return self.y_output


class ConvNetwork(nn.Module):
    '''Convolutional Q Network'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Conv2d(c.INPUT, 64, 2),
                                            nn.AvgPool2d(kernel_size=2),
                                            nn.Conv2d(2, 64, 2),
                                            nn.AvgPool2d(kernel_size=2),
                                            nn.Flatten(),
                                            nn.Linear(64, 64),
                                            nn.ReLU(),
                                            nn.Linear(64, 16),
                                            nn.ReLU(),
                                            nn.Linear(64, c.OUTPUT))
        self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        self.y_output = self.stacked_layers(x_input)
        return self.y_output
