from collections import namedtuple, deque
from itertools import islice
import random
import torch
from torch import  optim
from torch import nn
from torch.nn import functional as F
import config as c


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))


def dqn_optimization_step(network_model,
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

    for name, param in network_param_difference(network_model, source_network, transfer_learning):

        if l1_reg is None and 'weight' in name:
            l1_reg = param.norm(1)

        elif 'weight' in name:
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


def a2c_optimization(network_policy,
                     network_value_function,
                     source_value_function,
                     acc_batch,
                     q_values,
                     entr_coef=0,
                     learning_rate=0.0001,
                     lamb_lasso=0,
                     lamb_ridge=0,
                     momentum=0,
                     transfer_learning=False,
                     transfer='lasso'):
    '''Optimization step for A2C algorithm.'''

    criterion = nn.MSELoss()
    policy_optimizer = optim.Adam(network_policy.parameters(),
                                  lr=learning_rate,
                                  amsgrad=True)
    value_optimizer = optim.Adam(network_value_function.parameters(),
                                 lr=learning_rate,
                                 amsgrad=True)
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    batch = Transition(*zip(*acc_batch))
    state_batch = torch.stack(batch.state).to(c.DEVICE)
    action_batch = torch.tensor(batch.action).to(c.DEVICE)[:, None]

    if transfer == 'lasso':
        l1_reg = None

        for name, param in network_param_difference(network_policy, source_value_function, transfer_learning):

            if l1_reg is None and 'layer_res.weight' in name:  #layer_res.weight
                l1_reg = param.norm(1)

            elif 'layer_res.weight' in name:
                l1_reg = l1_reg + param.norm(1)

            if l1_reg is None and 'firstlayer.weight' in name:
                l1_reg = param.norm(1)

            elif 'firstlayer.weight' in name:
                l1_reg = l1_reg + param.norm(1)
            
            if l1_reg is None and 'secondlayer.weight' in name:
                l1_reg = param.norm(1)

            elif 'secondlayer.weight' in name:
                l1_reg = l1_reg + param.norm(1)
            
        
        reg_param = l1_reg

    elif transfer == 'group_lasso':
        group_lasso_reg = None

        for name, param in network_param_difference(network_policy, source_value_function, transfer_learning):

            if name in network_policy.transfer_kernels + network_policy.transfer_linear + network_policy.transfer_bias:

                if group_lasso_reg is None:
                    group_lasso_reg = param.norm(2)

                else:
                    group_lasso_reg = group_lasso_reg + param.norm(2)
        
        reg_param = group_lasso_reg
    
    elif transfer == 'group_lasso_kernel':
        
        group_lasso_reg = None

        for name, param in network_param_difference(network_policy, source_value_function, transfer_learning):

            if name in network_policy.transfer_kernels:
                total_kernel_reg = 0

                for i in range(param.shape[0]):
                    total_kernel_reg += param[i].norm(2)

                if group_lasso_reg is None:
                    group_lasso_reg = total_kernel_reg

                else:
                    group_lasso_reg += total_kernel_reg
            
            if name in network_policy.transfer_linear:
                if group_lasso_reg is None:
                    group_lasso_reg = param.norm(2)

                else:
                    group_lasso_reg = group_lasso_reg + param.norm(2)
            
            if name in network_policy.transfer_bias:
                if group_lasso_reg is None:
                    group_lasso_reg = param.norm(2)

                else:
                    group_lasso_reg = group_lasso_reg + param.norm(2)
        
        reg_param = group_lasso_reg
    
    else:
        reg_param = 0

    l_entr = torch.distributions.Categorical(logits=torch.nn.functional.log_softmax(network_policy(state_batch), dim=1))
    entropy = l_entr.entropy().mean()
    value_loss = (q_values.detach() - network_value_function(state_batch)[:, 0]).pow(2).mean() # + lamb_lasso*l1_reg + lamb_ridge*l2_reg
    value_loss.backward()
    value_optimizer.step()
    policy_loss = (-torch.nn.functional.log_softmax(network_policy(state_batch),
                                                    dim=1)[torch.arange(len(acc_batch)),
                                                           action_batch[:, 0]]*(q_values.detach() -
                                                                                network_value_function(state_batch)[:, 0].detach())).mean()+ lamb_lasso*reg_param #+ entr_coef*entropy
    policy_loss.backward()
    policy_optimizer.step()
    return value_loss.detach().cpu().numpy(), policy_loss.detach().cpu().numpy()

def network_param_difference(target_net, source_net, transfer_learning):
    '''Calculates difference between two networks' parameters for biased regularization'''

    regularization_vector = []

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        if transfer_learning:
            try:
                regularization_vector = param_target[1] - param_source[1]
            except:
                regularization_vector = param_target[1]
                print("Parameter mismatch in transfer learning. Likely incomatible dimensions in the last layer.")

        else:
            regularization_vector = param_target[1]

        yield param_target[0], regularization_vector


def calculate_q_value_batch(term_batch, network_value_function, reward_batch, last_state, time_steps, n_envs, gamma=0.9):
    '''Calculate the q value for the batch of states and rewards.'''

    q_values = torch.zeros(time_steps, n_envs).to(c.DEVICE)

    for i in range(n_envs):
        big_r = term_batch[-1][i]*network_value_function(last_state)[:, 0][i]
        for j in range(len(term_batch)):
            if term_batch[-j-1][i] == 0:
                big_r = 0
            big_r = gamma*big_r + reward_batch[-j-1][i]
            q_values[-j-1][i] = big_r

    return torch.flatten(q_values)
