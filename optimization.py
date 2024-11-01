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
                      memory_sample,
                      gamma,
                      learning_rate,
                      momentum=0.9,
                      lamb_lasso=1.0,
                      lamb_ridge=1.0,
                      batch_size=32,
                      source_network=None,
                      transfer_learning=False):
    "Optimization step given model and collected data."

    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(network_model.parameters(),
                           lr=learning_rate,
                           amsgrad=True,
                           betas=(momentum, momentum))
    # optimizer = optim.SGD(network_model.parameters(),
    #                       lr=learning_rate,
    #                       momentum=momentum)
    optimizer.zero_grad()
    set_size = len(memory_sample)

    if set_size < batch_size:
        return

    random.shuffle(memory_sample)
    set_size = batch_size
    batch_size = round(set_size)
    mem_batches = [deque(islice(memory_sample,
                                batch_size*i,
                                batch_size*(i + 1))) for i in range(1)]

    for mem_batch in mem_batches:
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
        loss_output = criterion(target_batch, q_output) + lamb_lasso*l1_reg + lamb_ridge*l2_reg
        pre_opt_loss = loss_output.to('cpu').detach().numpy()
        loss_output.backward()
        optimizer.step()

    return loss_value.to('cpu').detach().numpy(), pre_opt_loss
        # print("optimization step concluded")


def network_param_difference(target_net, source_net, transfer_learning):
    '''Calculates difference between two networks' parameters for biased regularization'''

    target_vector = []
    source_vector = []
    regularization_vector = []

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        if transfer_learning:
            regularization_vector = param_target[1] - param_source[1]

        else:
            regularization_vector = param_target[1]

        yield param_target[0], regularization_vector
