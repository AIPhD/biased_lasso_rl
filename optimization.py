from collections import namedtuple
import random
import torch
from torch import  optim
from torch import nn
import config as c


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class NetworkOptimizer():
    '''Optimizer given a neural network as input. Updates NN depending on predefined Loss.'''

    def __init__(self, network_model):
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(network_model.parameters(),
                                   lr=0.001,
                                   momentum=0.9)
        self.network_model = network_model

    def optimization_step(self, memory_sample):
        '''Calculates loss gradient and performs update on NN.'''
        if len(memory_sample) < c.BATCH_SIZE:
            return
        transition_sample = random.sample(memory_sample, c.BATCH_SIZE)
        batch = Transition(*zip(*transition_sample))
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action)[:, None]
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.tensor(batch.reward)
        q_output = self.network_model(state_batch).gather(1, action_batch).flatten()
        y_batch = reward_batch + c.GAMMA * torch.max(self.network_model(next_state_batch),
                                                     1).values
        loss = nn.MSELoss()
        loss_output = loss(y_batch, q_output)
        loss_output.backward()
        self.optimizer.step()
