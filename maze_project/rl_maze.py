import sys
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f
from collections import deque, namedtuple
import gymnasium as gym
import random
from stable_baselines3.common.vec_env import SubprocVecEnv
sys.path.append("/home/steven/biased_lasso_rl/gym-examples/")
import gym_examples
from maze_project import config_maze as cm
from maze_project import task_dict
import helper_functions as hf


class MazeA2CLearning():
    '''A2C Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=5000, lamb_lasso=0.1, learning_rate=1e-4,
                 batch_size=20, time_steps=10000, grid_size=7, n_envs = 16,
                 stochastic=True, source_network=None, transfer=None):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game = task_dict.GAME_TASKS[game_name]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.lamb_lasso = lamb_lasso
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.grid_size = grid_size
        self.stochastic = stochastic
        self.source_network = source_network
        self.transfer = transfer
        self.n_envs = n_envs

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env = gym.make('gym_examples/GridWorld-v0', game=self.game_name, render_mode='rgb_array')
        no_actions = env.action_space.n

        if self.source_network is not None:
            actor = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            actor.load_state_dict(torch.load(self.source_network))
            critic = MazeNetwork(self.grid_size, 1).to(cm.DEVICE)
            critic.load_state_dict(torch.load(self.source_network))
            actor.re_init_head()
            critic.re_init_head()

        else:
            actor = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            critic = MazeNetwork(self.grid_size, 1).to(cm.DEVICE)

        obs, _ = env.reset()
        state = create_maze_state(obs, size=self.grid_size)

        for i in range(self.time_steps):
            sample = []
            env.render()
            model_output = actor(torch.from_numpy(state)).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon, stochastic=True)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc, _ = env.step(action)
            next_state = create_maze_state(next_obs, self.grid_size)
            sample.append(hf.Transition(state, action, next_state, reward, term))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(actor, critic, sample)

            if term or trunc:
                obs, _ = env.reset()
                state = create_maze_state(obs, self.grid_size)
                sample = []
        
        env.close()
            

    def select_action(self, policy_output, epsilon):
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

    def optimize_model(self, actor, critic, sample):
        '''Optimization step for DQN algorithm.'''
        
        loss = nn.MSELoss()
        actor_optimizer = optim.Adam(actor.parameters(),
                                     lr=self.learning_rate,
                                     amsgrad=True)
        critic_optimizer = optim.Adam(critic.parameters(),
                                     lr=self.learning_rate,
                                     amsgrad=True)
        actor_optimizer.zero_grad()
        critic_optimizer.zero_grad()
        states, actions, next_states, rewards, term_bools = hf.create_batches(sample)
        critic_out = critic(states)
        critic_target = hf.calculate_q_value_batch_a2c(term_bools, critic, rewards, next_states[-1],
                                                       self.n_envs, self.gamma)
        critic_loss = loss(critic_out, critic_target.detach())
        actor_loss = (f.log_softmax(actor(states),
                                    dim=1).gather(1, actions[:, 0])*(critic_target -
                                                                     critic_out(states)[:, 0]).detach()).mean()
        reg_param = 0

        if self.transfer is not None and self.source_network is not None:
            reg_param = hf.biased_lasso_regularization(actor, self.source_network)


        critic_loss += self.lamb_lasso*reg_param
        actor_loss.backward()
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        actor_optimizer.step()
        actor_optimizer.zero_grad()


class MazeDQNLearning():
    '''DQN Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=5000, lamb_lasso=0.1, learning_rate=1e-4,
                 update_target=1000, batch_size=20, time_steps=10000, capacity=2000, grid_size=7,
                 stochastic=True, source_network=None, transfer=None):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game = task_dict.GAME_TASKS[game_name]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.lamb_lasso = lamb_lasso
        self.learning_rate = learning_rate
        self.update_target = update_target
        self.batch_size = batch_size
        self.time_steps = time_steps
        self.grid_size = grid_size
        self.stochastic = stochastic
        self.source_network = source_network
        self.transfer = transfer
        self.memory = deque([], maxlen=capacity)

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env = gym.make('gym_examples/GridWorld-v0', game=self.game, render_mode='rgb_array')
        no_actions = env.action_space.n

        if self.source_network is not None:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.load_state_dict(torch.load(self.source_network))
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target.load_state_dict(torch.load(self.source_network))
            model.re_init_head()
            target.re_init_head()

        else:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)

        obs, _ = env.reset()
        state = create_maze_state(obs, self.grid_size)

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(reward)
            next_state = create_maze_state(next_obs, self.grid_size)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.update_target) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = model.state_dict()[key]

            if term or trunc:
                obs, _ = env.reset()
                state = create_maze_state(obs, self.grid_size)
        
        env.close()
            

    def select_action(self, policy_output, epsilon):
        '''Select action based on epsilon-greedy policy.'''
        
        threshold = random.random()

        if threshold > epsilon:

            if self.stochastic:
                probabilities = f.softmax(policy_output, dim=0)
                action = np.random.choice(np.arange(len(probabilities)), p=probabilities.cpu().detach().numpy())

            else:
                action = torch.argmax(policy_output).cpu().detach().numpy()
        
        else:
            action = np.random.randint(4)
        
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
        states, actions, next_states, rewards, term_bools = hf.create_batches(sample)
        q_output = model(states).gather(1, actions[:, None])[:, 0]
        target_batch = rewards + self.gamma * term_bools * torch.max(target(next_states), 1)[0]
        total_loss = loss(q_output, target_batch.detach())
        reg_param = 0

        if self.transfer is not None and self.source_network is not None:
            reg_param = hf.biased_lasso_regularization(model, self.source_network)


        total_loss += self.lamb_lasso * reg_param
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class MazeDDQNLearning():
    '''Double DQN Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=1, eps_min=0.05, eps_decay=5000, random_phase=1000, lamb_lasso=0.1,
                 learning_rate=1e-4, batch_size=256, target_update=100, tau=1, time_steps=20000, capacity=10000, grid_size=5,
                 stochastic=False, source_network=None, transfer=None):
        '''Initialize the Double DQN agent with environment and hyperparameters.'''

        self.game = task_dict.GAME_TASKS[game_name]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_min
        self.eps_decay = eps_decay
        self.random_phase = random_phase
        self.lamb_lasso = lamb_lasso
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.target_update = target_update
        self.tau = tau
        self.time_steps = time_steps
        self.grid_size = grid_size
        self.stochastic = stochastic
        self.source_network = source_network
        self.transfer = transfer
        self.memory = deque([], maxlen=capacity)

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env = gym.make('gym_examples/GridWorld-v0', game=self.game,
                       render_mode='rgb_array', size=self.grid_size)
        no_actions = env.action_space.n

        if self.source_network is not None:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.load_state_dict(torch.load(self.source_network))
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target.load_state_dict(torch.load(self.source_network))
            model.re_init_head()
            target.re_init_head()

        else:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)

        obs, _ = env.reset()
        state = create_maze_state(obs, self.grid_size)

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(action, reward)

            if i > self.random_phase:
                self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)

            next_state = create_maze_state(next_obs, self.grid_size)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.target_update) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = self.tau*model.state_dict()[key] + (1-self.tau)*target.state_dict()[key]

            if term or trunc:
                obs, _ = env.reset()
                state = create_maze_state(obs, self.grid_size)
        
        env.close()
            

    def select_action(self, policy_output, epsilon):
        '''Select action based on epsilon-greedy policy.'''
        
        threshold = random.random()

        if threshold > epsilon:

            if self.stochastic:
                probabilities = f.softmax(policy_output, dim=0)
                action = np.random.choice(np.arange(len(probabilities)), p=probabilities.cpu().detach().numpy())

            else:
                action = int(torch.argmax(policy_output).cpu().detach().numpy())
        
        else:
            action = np.random.randint(4)
        
        return action

    def optimize_model(self, model, target):
        '''Optimization step for Double DQN algorithm.'''
        
        loss = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate)
        optimizer.zero_grad()
        sample = random.sample(self.memory, self.batch_size)
        states, actions, next_states, rewards, term_bools = hf.create_batches(sample)
        q_output = model(states).gather(1, actions[:, None])[:, 0]
        actions_model1 = torch.argmax(model(next_states), 1)
        target_batch = rewards + self.gamma * term_bools * target(next_states).gather(1, actions_model1[:, None])[:, 0]
        total_loss = loss(q_output, target_batch.detach())
        reg_param = 0

        if self.transfer is not None and self.source_network is not None:
            reg_param = hf.biased_lasso_regularization(model, self.source_network)


        total_loss += self.lamb_lasso * reg_param
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


class MazePPOLearning():
    pass


class MazeNetwork(nn.Module):
    '''Deep Q-Network Architecture.'''

    def __init__(self, grid_size, out_dim):

        super().__init__()
        self.out_dim = out_dim
        self.layer1 = nn.Linear(grid_size**2*5, 50 * grid_size**2)
        self.layer2 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer3 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer4 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer5 = nn.Linear(50 * grid_size**2, 50 * grid_size**2)
        self.layer6 = nn.Linear(50 * grid_size**2, out_dim)
        self.layerres1 = nn.Linear(grid_size**2*5, out_dim)
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
        self.layer6.weight.data = torch.zeros(self.out_dim, 50 * self.grid_size**2).to(cm.DEVICE)
        self.layer6.bias.data = torch.zeros(self.out_dim).to(cm.DEVICE)


def create_maze_state(observation, size):
    '''Convert observation value of maze environment to state vector'''
    state_vector = torch.zeros(5, size, size).to(cm.DEVICE)
    state_vector[0, observation['agent'][0], observation['agent'][1]] = 1

    for target in observation['target']:
        state_vector[1, target[0], target[1]] = 1

    for coin in observation['coins']:
        state_vector[2, coin[0], coin[1]] = 1

    for trap in observation['traps']:
        state_vector[3, trap[0], trap[1]] = 1

    for wall in observation['walls']:
        state_vector[4, wall[0], wall[1]] = 1

    state_vector = torch.flatten(state_vector)

    return state_vector
