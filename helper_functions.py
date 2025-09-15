import json
from collections import namedtuple
import torch
import config as c

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

def network_param_difference(target_net, source_net):
    '''Calculates difference between two networks' parameters for biased regularization'''

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        try:
            regularization_vector = param_target[1] - param_source[1]
        except:
            regularization_vector = param_target[1]
            print("Parameter mismatch in transfer learning. Likely incomatible dimensions in the last layer.")

        yield param_target[0], regularization_vector


def biased_lasso_regularization(target_net, source_net):

    l1_reg = None

    for name, param in network_param_difference(target_net, source_net):

        if name in target_net.transfer_layers:

            if l1_reg is None:
                l1_reg = param.norm(1)

            else:
                l1_reg += param.norm(1)
    
    return l1_reg


def create_batches(sample):

    batch = Transition(*zip(*sample))
    states = torch.stack(batch.state).to(c.DEVICE)
    actions = torch.tensor(batch.action).to(c.DEVICE)
    next_states = torch.stack(batch.next_state).to(c.DEVICE)
    rewards = torch.tensor(batch.reward).to(c.DEVICE)
    term_bools = torch.ones(len(sample)).to(c.DEVICE) - torch.tensor(batch.terminated).to(c.DEVICE)
    return states, actions, next_states, rewards, term_bools


def calculate_q_value_batch_a2c(term_batch, critic, reward_batch, last_state, n_envs, gamma=0.9):
    '''Calculate the q value for the batch of states and rewards.'''

    q_values = torch.zeros(len(term_batch), n_envs).to(c.DEVICE)

    for i in range(n_envs):
        big_r = term_batch[-1][i]*critic(last_state)[:, 0][i]
        for j in range(len(term_batch)):
            big_r = gamma*term_batch[-j-1][i]*big_r + reward_batch[-j-1][i]
            q_values[-j-1][i] = big_r

    return torch.flatten(q_values)

def import_json(file):
    pass
