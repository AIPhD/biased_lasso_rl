# import torch
from torch import nn
import config as c


class FullConnectedNetwork(nn.Module):
    '''Fully connected network for CartPole'''

    def __init__(self):

        super(FullConnectedNetwork, self).__init__()
        # self.stacked_layers = nn.Sequential(nn.Linear(c.INPUT, 16),
        #                                     nn.ReLU(),
        #                                     # nn.Linear(256,64),
        #                                     # nn.ReLU(),
        #                                     nn.Linear(16, 16),
        #                                     nn.ReLU(),
        #                                     nn.Linear(16, c.OUTPUT))
        # self.y_output = torch.Tensor(np.zeros(c.OUTPUT))
        self.layer1 = nn.Linear(c.CART_INPUT, c.HIDDEN_NODE_COUNT)
        self.layer2 = nn.Linear(c.HIDDEN_NODE_COUNT, c.HIDDEN_NODE_COUNT)
        self.layer3 = nn.Linear(c.HIDDEN_NODE_COUNT, c.CART_OUTPUT)


    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input = nn.functional.relu(self.layer1(x_input))
        x_input = nn.functional.relu(self.layer2(x_input))
        return self.layer3(x_input)


class ConvNetwork(nn.Module):
    '''Convolutional Q Network for mazes.'''

    def __init__(self):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Conv2d(3, 3, kernel_size=(2, 2)),
                                            nn.ReLU(),
                                            # nn.Conv2d(3, 3, kernel_size=(2, 2)),
                                            # nn.ReLU(),
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
                                            nn.Linear(8, c.GRID_OUTPUT))
#         self.y_output = torch.Tensor(np.zeros(c.OUTPUT))

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        y_output = self.stacked_layers(x_input)
        return y_output

class MazeFCNetwork(nn.Module):
    '''Convolutional Q Network for mazes.'''

    def __init__(self):

        super().__init__()

        self.layer1 = nn.Linear(c.GRID_INPUT, c.HIDDEN_NODE_COUNT)
        self.layer2 = nn.Linear(c.HIDDEN_NODE_COUNT, c.HIDDEN_NODE_COUNT)
        self.layer3 = nn.Linear(c.HIDDEN_NODE_COUNT, c.HIDDEN_NODE_COUNT)
        self.layer4 = nn.Linear(c.HIDDEN_NODE_COUNT, c.HIDDEN_NODE_COUNT)
        self.layer5 = nn.Linear(c.HIDDEN_NODE_COUNT, c.HIDDEN_NODE_COUNT)
        self.layer6 = nn.Linear(c.HIDDEN_NODE_COUNT, c.GRID_OUTPUT)

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''
        x_input = nn.functional.relu(self.layer1(x_input))
        x_input = nn.functional.relu(self.layer2(x_input))
        x_input = nn.functional.relu(self.layer3(x_input))
        x_input = nn.functional.relu(self.layer4(x_input))
        x_input = nn.functional.relu(self.layer5(x_input))
        return self.layer6(x_input)
