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
                                            # nn.Linear(256,64),
                                            # nn.ReLU(),
                                            nn.Linear(64, 64),
                                            nn.ReLU(),
                                            nn.Linear(16, c.OUTPUT))
        self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        self.y_output = self.stacked_layers(x_input)
        return self.y_output


class ConvNetwork(nn.Module):
    '''Convolutional Q Network'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Conv2d(3, 3, kernel_size=(3, 3)),
                                            nn.ReLU(),
                                            nn.Conv2d(3, 3, kernel_size=(2, 2)),
                                            nn.ReLU(),
                                            nn.MaxPool2d(kernel_size=3),
                                            # nn.Conv2d(4, 4, kernel_size=(3, 3)),
                                            # nn.ReLU(),
                                            # nn.Conv2d(3, 3, kernel_size=(2, 2)),
                                            # nn.ReLU(),
                                            # nn.MaxPool2d(kernel_size=2),
                                            # nn.Conv2d(3, 3, kernel_size=(2, 2)),
                                            # nn.AvgPool2d(kernel_size=2),
                                            nn.Flatten(start_dim=1),
                                            nn.Linear(12, 8),
                                            nn.ReLU(),
                                            # nn.Linear(36, 12),
                                            # nn.ReLU(),
                                            nn.Linear(8, c.OUTPUT))
#         self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        y_output = self.stacked_layers(x_input)
        return y_output
