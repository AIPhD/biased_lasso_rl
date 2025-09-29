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
from gymnasium.wrappers import AtariPreprocessing, FrameStackObservation
from stable_baselines3.common.vec_env import SubprocVecEnv
sys.path.append("/home/steven/biased_lasso_rl/gym-examples/")
import gym_examples
from atari_project import config_atari as ca
from atari_project import atari_dict
import helper_functions as hf


class AtariA2CLearning():
    '''A2C Agent for different Atari tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=500, lamb_lasso=0.1, learning_rate=1e-4,
                 batch_size=20, time_steps=10000, grid_size=7, n_envs = 16, frame_skip=4,
                 stochastic=True, source_network=None, transfer=None):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game = atari_dict.ATARI_GAMES[game_name]
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
        self.frame_skip = frame_skip


    def make_env(self, seed, game):
        def _init():
            env = gym.make(game,
                           render_mode='rgb_array',
                           frameskip=1)
            env = AtariPreprocessing(env,
                                     noop_max=30,
                                     frame_skip=self.frame_skip,
                                     screen_size=84,
                                     terminal_on_life_loss=False,
                                     grayscale_obs=True,
                                     grayscale_newaxis=False,
                                     scale_obs=False)
            env = FrameStackObservation(env, 4)
            env.reset(seed=seed)
            return env
        return _init


    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env_fns = [self.make_env(i, self.game) for i in range(self.n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        no_actions = vec_env.action_space.n

        if self.source_network is not None:
            actor = AtariNetwork(self.grid_size, no_actions).to(ca.DEVICE)
            actor.load_state_dict(torch.load(self.source_network))
            actor.re_init_head()

        else:
            actor = AtariNetwork(no_actions).to(ca.DEVICE)
            actor.initialize_network()

        critic = AtariNetwork(1).to(ca.DEVICE)    
        critic.initialize_network()
        obs = vec_env.reset()
        state = create_conv_state_vector_mult_proc(obs, self.n_envs)
        sample = []

        for i in range(self.time_steps):
            vec_env.render()
            model_output = actor(state).to(ca.DEVICE)
            actions = self.select_action(model_output, self.epsilon)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc = vec_env.step(actions)

            for i in range(self.n_envs):

                if trunc[i]['TimeLimit.truncated'] or bool(term[i]):
                    next_state = create_conv_state_vector(trunc[i]['terminal_observation'])

                else:
                    next_state = create_conv_state_vector_mult_proc(next_obs,  self.n_envs)[i]

                    sample.append(hf.Transition(state[i],
                                                actions[i],
                                                next_state,
                                                reward[i],
                                                term[i]))

            state = create_conv_state_vector_mult_proc(next_obs,  self.n_envs)

            if ((i+1) % self.batch_size) == 0:
                self.optimize_model(actor, critic, sample)
                sample = []
        
        vec_env.close()
            

    def select_action(self, policy_output, epsilon):
        '''Select action based on epsilon-greedy policy.'''
        
        thresholds = np.random.sample(size=self.n_envs)
        actions = np.zeros(self.n_envs, dtype=int)

        if torch.max(policy_output).isnan():
                    print('Nan Value detected')
    
        for i in range(self.n_envs):
            if thresholds[i] > epsilon:

                if self.stochastic:
                    probabilities = f.softmax(policy_output[i], dim=0)
                    actions[i] = np.random.choice(np.arange(len(probabilities)), p=probabilities.cpu().detach().numpy())

                else:
                    actions[i] = torch.argmax(policy_output[i]).cpu().detach().numpy()
            
            else:
                actions[i] = np.random.randint(4)
        
        return actions

    def optimize_model(self, actor, critic, sample):
        '''Optimization step for A2C algorithm.'''
        
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
        critic_target = hf.calculate_q_target_batch(term_bools, critic, rewards, next_states[-self.n_envs:],
                                                    self.batch_size, self.n_envs, self.gamma)
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


class AtariDQNLearning():
    '''DQN Agent for different Atari games, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=5000, lamb_lasso=0.1, learning_rate=1e-4,
                 update_target=1000, batch_size=20, time_steps=10000, capacity=2000, grid_size=7,
                 stochastic=True, source_network=None, transfer=None):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game = atari_dict.ATARI_GAMES[game_name]
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

        env = gym.make(self.game, render_mode='rgb_array')
        no_actions = env.action_space.n

        if self.source_network is not None:
            model = AtariNetwork(no_actions).to(ca.DEVICE)
            model.load_state_dict(torch.load(self.source_network))
            target = AtariNetwork(no_actions).to(ca.DEVICE)
            target.load_state_dict(torch.load(self.source_network))
            model.re_init_head()
            target.re_init_head()

        else:
            model = AtariNetwork(no_actions).to(ca.DEVICE)
            target = AtariNetwork(no_actions).to(ca.DEVICE)
            model.initialize_network()
            target.initialize_network()

        obs, _ = env.reset()
        state = create_conv_state_vector(obs)

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(ca.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(reward)
            next_state = create_conv_state_vector(next_obs)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.update_target) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = model.state_dict()[key]

            if term or trunc:
                obs, _ = env.reset()
                state = create_conv_state_vector(obs)
        
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


class AtariDDQNLearning():
    '''Double DQN Agent for different Atari tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=1, eps_min=0.05, eps_decay=5000, random_phase=1000, lamb_lasso=0.1,
                 learning_rate=1e-4, batch_size=256, target_update=100, tau=1, time_steps=20000, capacity=10000, grid_size=5,
                 stochastic=False, source_network=None, transfer=None):
        '''Initialize the Double DQN agent with environment and hyperparameters.'''

        self.game = atari_dict.ATARI_GAMES[game_name]
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

        env = gym.make(game=self.game,
                       render_mode='rgb_array')
        no_actions = env.action_space.n

        if self.source_network is not None:
            model = AtariNetwork(no_actions).to(ca.DEVICE)
            model.load_state_dict(torch.load(self.source_network))
            target = AtariNetwork(no_actions).to(ca.DEVICE)
            target.load_state_dict(torch.load(self.source_network))
            model.re_init_head()
            target.re_init_head()

        else:
            model = AtariNetwork(no_actions).to(ca.DEVICE)
            target = AtariNetwork(no_actions).to(ca.DEVICE)
            model.initialize_network()
            target.initialize_network()

        obs, _ = env.reset()
        state = create_conv_state_vector(obs)

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(ca.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(action, reward)

            if i > self.random_phase:
                self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)

            next_state = create_conv_state_vector(next_obs)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.target_update) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = self.tau*model.state_dict()[key] + (1-self.tau)*target.state_dict()[key]

            if term or trunc:
                obs, _ = env.reset()
                state = create_conv_state_vector(obs)
        
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


class AtariPPOLearning():
    pass

class AtariNetwork(nn.Module):
    '''Convolutional Net structure for Atari game environments'''

    def __init__(self, num_actions):

        super().__init__()
        self.firstlayer = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.secondlayer = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.thirdlayer = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fourthlayer = nn.Linear(3136, 1600)
        self.fifthlayer = nn.Linear(1600, 256)
        self.sixthlayer = nn.Linear(256, num_actions)
        self.num_actions = num_actions
        self.transfer_kernels = ['firstlayer.weight',
                                 'secondlayer.weight',
                                 'thirdlayer.weight']
        self.transfer_bias = ['firstlayer.bias',
                              'secondlayer.weight',
                              'thirdlayer.bias',
                              'fourthlayer.bias']
        self.transfer_linear = ['fourthlayer.weight']

    def forward(self, x_input):
        '''Calculates output of the network given data x_input'''

        x_input = self.firstlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.secondlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.thirdlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = torch.flatten(x_input, start_dim=1)
        x_input = self.fourthlayer(x_input)
        x_input = nn.functional.relu(x_input)
        x_input = self.fifthlayer(x_input)
        x_input = nn.functional.relu(x_input)
        y_output = self.sixthlayer(x_input)
        return y_output
    
    def initialize_network(self):
        torch.nn.init.kaiming_uniform_(self.firstlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.firstlayer.bias)
        torch.nn.init.kaiming_uniform_(self.secondlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.secondlayer.bias)
        torch.nn.init.kaiming_uniform_(self.thirdlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.thirdlayer.bias)
        torch.nn.init.kaiming_uniform_(self.fourthlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fourthlayer.bias)
        torch.nn.init.kaiming_uniform_(self.fifthlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fifthlayer.bias)
        torch.nn.init.xavier_uniform_(self.sixthlayer.weight)
        torch.nn.init.zeros_(self.sixthlayer.bias)
    
    def re_init_head(self):
        torch.nn.init.kaiming_uniform_(self.fifthlayer.weight, nonlinearity='relu')
        torch.nn.init.zeros_(self.fifthlayer.bias)
        torch.nn.init.xavier_uniform_(self.sixthlayer.weight)
        torch.nn.init.zeros_(self.sixthlayer.bias)

    def copy_layers(self, source_model_dict):
        for key in source_model_dict:
            try:
                self.state_dict()[key] = source_model_dict[key]
            except:
                print('incompatible_layer')


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 4, 84, 84).to(ca.DEVICE)
    state_vector[0] = torch.from_numpy(np.reshape(observation, (4, 84, 84)))
    return state_vector


def create_conv_state_vector_mult_proc(observations, n_envs):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(n_envs, 4, 84, 84).to(ca.DEVICE)
    for i in range(n_envs):
        state_vector[i] = torch.from_numpy(np.reshape(observations[i], (4, 84, 84)))
    return state_vector
