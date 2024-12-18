from collections import namedtuple, deque
from itertools import islice
import random
import torch
from torch import  optim
from torch import nn
import config as c


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))


def optimization_step(network_model,
                      target_net,
                      memory,
                      gamma,
                      learning_rate,
                      momentum=0.9,
                      lamb_lasso=1.0,
                      lamb_ridge=1.0,
                      batch_size=32,
                      source_network=None,
                      transfer_learning=False):
    "Optimization step given model and collected data."

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network_model.parameters(),
                           lr=learning_rate,
                           amsgrad=True,
                           betas=(momentum, momentum))
    # optimizer = optim.SGD(network_model.parameters(),
    #                       lr=learning_rate,
    #                       momentum=momentum)
    optimizer.zero_grad()
    # set_size = len(memory_sample)

    # if set_size < batch_size:
    #     return

    # random.shuffle(memory_sample)
    # set_size = batch_size
    # batch_size = round(set_size)
    # mem_batches = [deque(islice(memory_sample,
    #                             batch_size*i,
    #                             batch_size*(i + 1))) for i in range(1)]
    mem_batch = random.sample(memory, batch_size)
    batch = Transition(*zip(*mem_batch))
    state_batch = torch.stack(batch.state).to(c.DEVICE)
    action_batch = torch.tensor(batch.action).to(c.DEVICE)[:, None]
    next_state_batch = torch.stack(batch.next_state).to(c.DEVICE)
    reward_batch = torch.tensor(batch.reward).to(c.DEVICE)
    term_bool = torch.tensor(batch.terminated).to(c.DEVICE)
    q_output = network_model(state_batch).gather(1, action_batch).flatten()

    with torch.no_grad():
        target_batch = reward_batch + gamma*term_bool*torch.max(target_net(next_state_batch),
                                                                1).values

    l2_reg = None
    l1_reg = None

    # if transfer_learning:
    #     reg_vector = network_param_difference(network_model, source_network)

    # else:
    #     reg_vector = []

    #     for key in network_model.state_dict():
    #         reg_vector.append(network_model.state_dict()[key])

    for name, param in network_param_difference(network_model, source_network, transfer_learning):

        if l1_reg is None and 'weight' in name:
            # l2_reg = param.norm(2)**2
            l1_reg = param.norm(1)

        elif 'weight' in name:
            # l2_reg = l2_reg + param.norm(2)**2
            l1_reg = l1_reg + param.norm(1)

    for name, ridge_param in network_model.named_parameters():

        if l2_reg is None and 'weight' in name:
            l2_reg = ridge_param.norm(2)**2

        elif 'weight' in name:
            l2_reg = l2_reg + ridge_param.norm(2)**2

    loss_value = criterion(target_batch, q_output)
    loss_value_detached = loss_value.to('cpu').detach().numpy()
    loss_output = criterion(target_batch, q_output) + lamb_lasso*l1_reg + lamb_ridge*l2_reg
    pre_opt_loss = loss_output.to('cpu').detach().numpy()
    loss_output.backward()
    optimizer.step()
    optimizer.zero_grad()

    return loss_value_detached, pre_opt_loss
        # print("optimization step concluded")


def a3c_optimization(network_policy, network_value_function, acc_batch, big_r, gamma, learning_rate=1, momentum=0, transfer_learning=False):
    '''Optimization step for A3C algorithm.'''

    criterion = nn.MSELoss()
    policy_optimizer = optim.Adam(network_policy.parameters(),
                                  lr=1,
                                  amsgrad=True)
    value_optimizer = optim.Adam(network_value_function.parameters(),
                                 lr=1,
                                 amsgrad=True)
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    batch = Transition(*zip(*acc_batch))
    state_batch = torch.stack(batch.state).to(c.DEVICE)
    action_batch = torch.tensor(batch.action).to(c.DEVICE)[:, None]
    reward_batch = torch.tensor(batch.reward).to(c.DEVICE)

    for i in range(len(reward_batch)):
        big_r = gamma*big_r + reward_batch[-i-1]
        policy_loss = torch.log(network_policy(state_batch[-i-1])[action_batch[-i-1]])*(big_r - network_value_function(state_batch[-i-1]).values)
        value_loss = criterion(big_r, network_value_function(state_batch[-i-1]))
        policy_loss.backward()
        value_loss.backward()

    policy_optimizer.step()
    value_optimizer.step()

    return value_loss, policy_loss


def network_param_difference(target_net, source_net, transfer_learning):
    '''Calculates difference between two networks' parameters for biased regularization'''

    regularization_vector = []

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        if transfer_learning:
            regularization_vector = param_target[1] - param_source[1]

        else:
            regularization_vector = param_target[1]

        yield param_target[0], regularization_vector
