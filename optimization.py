from collections import namedtuple, deque
from itertools import islice
import random
import torch
from torch import  optim
from torch import nn
import config as c


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


def optimization_step(network_model,
                      target_net,
                      memory_sample,
                      no_segments,
                      gamma,
                      learning_rate,
                      lamb,
                      batch_size,
                      source_network=None,
                      transfer_learning=True,
                      mini_batch_training=c.MINI_BATCH_TRAINING):
    "Optimization step given model and collected data."

    criterion = nn.MSELoss()
    optimizer = optim.Adam(network_model.parameters(),
                           lr=learning_rate,
                           amsgrad=False)
    # optimizer = optim.SGD(network_model.parameters(),
    #                        lr=c.LEARNING_RATE)
    set_size = len(memory_sample)
    target_net.requires_grad_(requires_grad=False)

    if set_size < batch_size:
        return

    random.shuffle(memory_sample)

    if not mini_batch_training:
        no_segments = 1
        set_size = batch_size

    batch_size = round(set_size/no_segments)
    mem_batches = [deque(islice(memory_sample,
                                batch_size*i,
                                batch_size*(i + 1))) for i in range(no_segments)]

    for mem_batch in mem_batches:
        batch = Transition(*zip(*mem_batch))
        state_batch = torch.stack(batch.state)
        action_batch = torch.tensor(batch.action).to(c.DEVICE)[:, None]
        next_state_batch = torch.stack(batch.next_state)
        reward_batch = torch.tensor(batch.reward).to(c.DEVICE)
        q_output = network_model(state_batch).gather(1, action_batch).flatten()
        with torch.no_grad():
            target_batch = reward_batch + gamma*torch.max(target_net(next_state_batch),
                                                                1).values
        l2_reg = None
        l1_reg = None

        reg_vector = network_param_difference(target_net, source_network)

        for param in reg_vector:

            if l2_reg is None and l1_reg is None:
                l2_reg = param.norm(2)**2
                l1_reg = param.norm(1)

            else:
                l2_reg = l2_reg + param.norm(2)**2
                l1_reg = l1_reg + param.norm(1)

        loss_output = criterion(target_batch, q_output) + lamb*l1_reg
        loss_output.backward()
        optimizer.step()
        # print("optimization step concluded")


def network_param_difference(target_net, source_net):
    '''Calculates difference between two networks' parameters for biased regularization'''

    regularization_vector = []

    for key in target_net.state_dict():
            regularization_vector.append(target_net.state_dict()[key] - source_net.state_dict()[key])

    return regularization_vector
