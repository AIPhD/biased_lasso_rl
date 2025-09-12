import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from collections import deque, namedtuple
import gymnasium as gym
import random
from stable_baselines3.common.vec_env import SubprocVecEnv
from maze_project import config_maze as cm
import transfer_functions as tf

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminated'))

class MazeDQNAgent():
    '''DQN Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game, gamma=0.99, epsilon=0.9, lamb_lasso=0.1, learning_rate=1e-4,
                 update_target=1000, batch_size=20, time_steps=10000, grid_size=9,
                 stochastic=True, source_network=None, transfer=None):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game = game
        self.gamma = gamma
        self.epsilon = epsilon
        self.lamb_lasso = lamb_lasso
        self.learning_rate = learning_rate
        self.update_target = update_target
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.grid_size = grid_size
        self.stochastic = stochastic
        self.source_network = source_network
        self.transfer = transfer
        self.memory = deque([], maxlen=cm.CAPACITY)

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        if self.source_network is not None:
            model = DQNNetwork(self.grid_size).to(cm.DEVICE)
            model.load_state_dict(torch.load(self.source_network))
            target = DQNNetwork(self.grid_size).to(cm.DEVICE)
            target.load_state_dict(torch.load(self.source_network))
            model.re_init_head()
            target.re_init_head()

        else:
            model = DQNNetwork(self.grid_size).to(cm.DEVICE)
            target = DQNNetwork(self.grid_size).to(cm.DEVICE)

        env = gym.make('gym_examples/GridWorld-v0', game=self.game, render_mode='rgb_array')
        state = env.reset()

        for i in range(self.time_steps):
            env.render()
            model_output = model(torch.from_numpy(state)).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon, stochastic=True)
            next_state, reward, term, trunc, _ = env.step(action)
            self.memory.append(Transition(state, action, next_state, reward, term))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.update_target) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = model.state_dict()[key]

            if term or trunc:
                state = env.reset()
        
        env.close()
            

    def select_action(self, policy_output, epsilon, stochastic=True):
        '''Select action based on epsilon-greedy policy.'''
        
        threshold = random.random()

        if threshold > epsilon:

            if self.stochastic:
                probabilities = f.softmax(policy_output, dim=0)
                action = np.random.choice(np.arange(len(probabilities)), p=probabilities.cpu().detach().numpy())

            else:
                action = torch.argmax(policy_output).cpu().detach().numpy()
        
        else:
            action = random.randint(4)
        
        return action

    def optimize_model(self,
                       model,
                       target):
        '''Optimization step for DQN algorithm.'''
        
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate,
                               amsgrad=True)
        optimizer.zero_grad()
        sample = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*sample))
        states = torch.stack(batch.state).to(cm.DEVICE)
        actions = torch.tensor(batch.action).to(cm.DEVICE)
        next_states = torch.stack(batch.next_state).to(cm.DEVICE)
        rewards = torch.tensor(batch.reward).to(cm.DEVICE)
        term_bools = torch.ones(self.batch_size).to(cm.DEVICE) - torch.tensor(batch.terminated).to(cm.DEVICE)
        q_output = model(states).gather(1, actions[:, None])
        target_batch = rewards + self.gamma * term_bools * torch.max(target(next_states), 1)
        total_loss = loss(q_output, target_batch.detach())

        if self.transfer is not None and self.source_network is not None:
            l1_reg = None

            for name, param in tf.network_param_difference(model, self.source_network):

                if name in model.transfer_layers:

                    if l1_reg is None:
                        l1_reg = param.norm(1)

                    else:
                        l1_reg += param.norm(1)


        total_loss += self.lamb_lasso * l1_reg
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class DQNNetwork(nn.Module):
    '''Deep Q-Network Architecture.'''

    def __init__(self, grid_size):

        super().__init__()
        self.layer1 = nn.Linear(grid_size**2*5, 50 * grid_size**2)
        self.layer2 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer3 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer4 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer5 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer6 = nn.Linear(50 * grid_size**2, 4)
        self.layerres1 = nn.Linear(grid_size**2*5, 4)
        self.grid_size = grid_size
        self.transfer_layers = ['layer1.weight', 'layer2.weight', 'layer3.weight', 'layer4.weight',
                                'layer1.bias', 'layer2.bias', 'layer3.bias', 'layer4.bias',
                                'layerres1.weight', 'layerres1.bias']
    
    def forward(self, x_input):
        x = f.relu(self.layer1(x_input))
        x = f.relu(self.layer2(x))
        x = f.relu(self.layer3(x))
        x = f.relu(self.layer4(x))
        x = f.relu(self.layer5(x))
        x = self.layer6(x) + self.layerres1(x_input)
        return x
    
    def re_init_head(self):
        self.layer5.weight.data = torch.zeros(50 * self.grid_size**2, 50 * self.grid_size**2).to(cm.DEVICE)
        self.layer5.bias.data = torch.zeros(50 * self.grid_size**2).to(cm.DEVICE)
        self.layer6.weight.data = torch.zeros(4, 50 * self.grid_size**2).to(cm.DEVICE)
        self.layer6.bias.data = torch.zeros(4).to(cm.DEVICE)