from torch import  optim
from torch import nn


class NetworkOptimizer():
    '''Optimizer given a neural network as input. Updates NN depending on predefined Loss.'''

    def __init__(self, network_model):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(network_model.parameters(),
                                   lr=0.001,
                                   momentum=0.9)
        self.network_model = network_model

    def optimization_step(self, x_batch, y_batch):
        '''Calculates loss gradient and performs update on NN.'''
        y_output = self.network_model.forward(x_batch)
        loss = nn.MSELoss(y_batch, y_output)
        loss.backward()
        self.optimizer.step()
