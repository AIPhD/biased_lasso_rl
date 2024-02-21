import torch
from torch import nn
import numpy as np
import config as c


class ConvQNetwork(nn.Module):
    '''Convolutional Q Network'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Linear(c.INPUT, 512),
                                            nn.ReLU(),
                                            nn.Linear(512,512),
                                            nn.ReLU(),
                                            nn.Linear(512, c.OUTPUT))
        self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        self.y_output = self.stacked_layers(x_input)
        return self.y_output
