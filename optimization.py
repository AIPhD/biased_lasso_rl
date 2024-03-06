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
                      mini_batch_training=c.MINI_BATCH_TRAINING):
    "Optimization step given model and collected data."

    criterion = nn.MSELoss()
    optimizer = optim.SGD(network_model.parameters(),
                          lr=0.001,
                          momentum=0.9)
    set_size = len(memory_sample)

    if set_size < c.BATCH_SIZE:
        return

    random.shuffle(memory_sample)

    if not mini_batch_training:
        no_segments = 1
        set_size = c.BATCH_SIZE

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
            target_batch = reward_batch + c.GAMMA * torch.max(target_net(next_state_batch),
                                                                1).values
        loss_output = criterion(target_batch, q_output)
        loss_output.backward()
        optimizer.step()
