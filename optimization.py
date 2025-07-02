from collections import namedtuple, deque
from itertools import islice
import random
import torch
from torch import  optim
from torch import nn
from torch.nn import functional as F
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


def a2c_optimization(network_policy, network_value_function, source_value_function, acc_batch, big_r, gamma, learning_rate=1, lamb_lasso=0, lamb_ridge=0, momentum=0, transfer_learning=False):
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
    reward_batch = torch.tensor(batch.reward).to(c.DEVICE)
    term_batch = torch.tensor(batch.terminated).to(c.DEVICE)
    q_values = torch.zeros(len(acc_batch))
    l2_reg = None
    l1_reg = None

    for i in range(len(reward_batch)):
        big_r = gamma*big_r*term_batch[-i-1] + reward_batch[-i-1]
        q_values[-i-1] = big_r
        # policy_loss = torch.nn.functional.log_softmax(network_policy(state_batch[-i-1][None,:, :, :]),
        #                                               dim=1)[0, action_batch[-i-1][0]]*(big_r.detach() -
        #                                                                                 network_value_function(state_batch[-i-1][None, :, :, :]).detach())[0][0]
        # value_loss = criterion(big_r.detach()[0][0], network_value_function(state_batch[-i-1][None, :, :, :])[0][0])
        # policy_loss.backward()
        # value_loss.backward()  

    for name, param in network_param_difference(network_policy, source_value_function, transfer_learning):

        if l1_reg is None and 'layer_res.weight' in name:  #layer_res.weight
            l1_reg = param.norm(1)

        elif 'layer_res.weight' in name:
            l1_reg = l1_reg + param.norm(1)

    for name, ridge_param in network_value_function.named_parameters():

        if l2_reg is None and 'weight' in name:
            l2_reg = ridge_param.norm(2)**2

        elif 'weight' in name:
            l2_reg = l2_reg + ridge_param.norm(2)**2


    value_loss = (q_values.detach() - network_value_function(state_batch)[:, 0]).pow(2).mean() # + lamb_lasso*l1_reg + lamb_ridge*l2_reg
    value_loss.backward()
    value_optimizer.step()
    policy_loss = (-torch.nn.functional.log_softmax(network_policy(state_batch),
                                                    dim=1)[torch.arange(len(acc_batch)),
                                                           action_batch[:, 0]]*(q_values.detach() -
                                                                                network_value_function(state_batch)[:, 0].detach())).mean()+ lamb_lasso*l1_reg
    policy_loss.backward()
    policy_optimizer.step()

    return value_loss.detach().numpy(), policy_loss.detach().numpy()


def actor_critic_optimization(network_policy,
                     network_value_function,
                     reward,
                     action,
                     state,
                     next_state,
                     term,
                     gamma,
                     learning_rate=1e-3,
                     momentum=0,
                     transfer_learning=False):
    '''Optimization step for Actor_critic algorithm.'''

    criterion = nn.MSELoss()
    policy_optimizer = optim.Adam(network_policy.parameters(),
                                  lr=learning_rate,
                                  amsgrad=True)
    value_optimizer = optim.Adam(network_value_function.parameters(),
                                 lr=learning_rate,
                                 amsgrad=True)
    policy_optimizer.zero_grad()
    value_optimizer.zero_grad()
    value_loss = criterion(reward + term*gamma*network_value_function(next_state), network_value_function(state))
    value_loss.backward()
    value_optimizer.step()
    advantage_term = reward + term*gamma*network_value_function(next_state) - network_value_function(state)
    policy_loss = torch.nn.functional.log_softmax(network_policy(state), dim=1)[0, action]*advantage_term.detach()
    policy_loss.backward()
    policy_optimizer.step()

    return value_loss.detach().numpy(), policy_loss.detach().numpy()


def network_param_difference(target_net, source_net, transfer_learning):
    '''Calculates difference between two networks' parameters for biased regularization'''

    regularization_vector = []

    for param_target, param_source in zip(target_net.named_parameters(), source_net.named_parameters()):

        if transfer_learning:
            regularization_vector = param_target[1] - param_source[1]

        else:
            regularization_vector = param_target[1]

        yield param_target[0], regularization_vector


def prox(v, u, *, lambda_, lambda_bar, M):
    """
    v has shape (m,) or (m, batches)
    u has shape (k,) or (k, batches)

    supports GPU tensors
    """
    onedim = len(v.shape) == 1
    if onedim:
        v = v.unsqueeze(-1)
        u = u.unsqueeze(-1)

    u_abs_sorted = torch.sort(u.abs(), dim=0, descending=True).values

    k, batch = u.shape

    s = torch.arange(k + 1.0).view(-1, 1).to(v)
    zeros = torch.zeros(1, batch).to(u)

    a_s = lambda_ - M * torch.cat(
        [zeros, torch.cumsum(u_abs_sorted - lambda_bar, dim=0)]
    )

    norm_v = torch.norm(v, p=2, dim=0)

    x = F.relu(1 - a_s / norm_v) / (1 + s * M**2)

    w = M * x * norm_v
    intervals = soft_threshold(lambda_bar, u_abs_sorted)
    lower = torch.cat([intervals, zeros])

    idx = torch.sum(lower > w, dim=0).unsqueeze(0)

    x_star = torch.gather(x, 0, idx).view(1, batch)
    w_star = torch.gather(w, 0, idx).view(1, batch)

    beta_star = x_star * v
    theta_star = sign_binary(u) * torch.min(soft_threshold(lambda_bar, u.abs()), w_star)

    if onedim:
        beta_star.squeeze_(-1)
        theta_star.squeeze_(-1)

    return beta_star, theta_star


def soft_threshold(l, x):
    return torch.sign(x) * torch.relu(torch.abs(x) - l)


def sign_binary(x):
    ones = torch.ones_like(x)
    return torch.where(x >= 0, ones, -ones)
