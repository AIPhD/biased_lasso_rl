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

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=500, lamb_lasso=0.1, learning_rate=1e-4,
                 batch_size=20, time_steps=10000, grid_size=7, n_envs=16,
                 stochastic=True, source_name=None, model_dir='maze_project/saved_models/', data_dir='maze_project/data/'):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game_name = game_name
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
        self.source_name = source_name
        self.n_envs = n_envs
        self.model_dir = model_dir
        self.data_dir = data_dir


    def make_env(self, seed, game):
        def _init():
            '''Helper function to create multiple environments for parallel processing.'''
            env = gym.make('gym_examples/GridWorld-v0',
                            game=game,
                            render_mode='rgb_array',
                            size=self.grid_size)
            env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
            env.reset(seed=seed)
            return env
        return _init


    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env_fns = [self.make_env(i, self.game) for i in range(self.n_envs)]
        vec_env = SubprocVecEnv(env_fns)
        no_actions = vec_env.action_space.n

        if self.source_name is not None:
            actor = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            actor.load_state_dict(torch.load(self.model_dir + self.source_name + '_policy',
                                             map_location=torch.device(cm.DEVICE)))
            actor.re_init_head()
            source_network = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            source_network.load_state_dict(torch.load(self.model_dir + self.source_name + '_policy',
                                                      map_location=torch.device(cm.DEVICE)))
            source_network.eval()
            for param in source_network.parameters():
                param.requires_grad = False

        else:
            actor = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            actor.initialize_network()
            source_network = None

        critic = MazeNetwork(self.grid_size, 1).to(cm.DEVICE)    
        critic.initialize_network()
        obs = vec_env.reset()
        state = create_fc_state_vector_mult_proc(obs, size=self.grid_size, n_envs=self.n_envs)
        sample = []
        rewards = []

        for i in range(self.time_steps):
            vec_env.render()
            model_output = actor(state).to(cm.DEVICE)
            actions = self.select_action(model_output, self.epsilon)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc = vec_env.step(actions)

            for j in range(self.n_envs):

                if trunc[j]['TimeLimit.truncated'] or bool(term[j]):
                    next_state = create_maze_state(trunc[j]['terminal_observation'], self.grid_size)

                else:
                    next_state = create_fc_state_vector_mult_proc(next_obs, self.grid_size, self.n_envs)[j]

                sample.append(hf.Transition(state[j],
                                            actions[j],
                                            next_state,
                                            reward[j],
                                            int(term[j])))

            state = create_fc_state_vector_mult_proc(next_obs, self.grid_size, self.n_envs)
            rewards.append(reward)

            if ((i+1) % self.batch_size) == 0:
                self.optimize_model(actor, critic, source_network, sample)
                mean_cumulative_reward = np.sum(np.asarray(rewards), axis=0).mean()
                print(mean_cumulative_reward)
                rewards = []
                sample = []
            
            if ((i + 1) % self.batch_size*10) == 0:
                print('Saving Model and Rewards')
                self.save_model(actor, critic)
                # self.save_rewards(mean_cumulative_reward)
        
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

    def optimize_model(self, actor, critic, source, sample):
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
        critic_loss = loss(critic_out[:, 0], critic_target.detach())
        actor_loss = (-f.log_softmax(actor(states),
                                     dim=1).gather(1, actions[:, None])*(critic_target -
                                                                         critic_out[:, 0]).detach()).mean()
        reg_param = 0

        if source is not None:
            reg_param = hf.biased_lasso_regularization(actor, source)


        critic_loss += self.lamb_lasso*reg_param
        actor_loss.backward()
        critic_loss.backward()
        critic_optimizer.step()
        critic_optimizer.zero_grad()
        actor_optimizer.step()
        actor_optimizer.zero_grad()
    
    def save_model(self, actor, critic):
        '''Save models to a specified filename.'''
        torch.save(actor.state_dict(), self.model_dir + self.game_name + '_a2c_policy')
        torch.save(critic.state_dict(), self.model_dir + self.game_name + '_a2c_value')
    
    def save_rewards(self, rewards):
        '''Save the rewards to a specified filename.'''
        np.save(self.data_dir + self.game_name + f'_source_{self.source_name}_lamb_{self.lamb_lasso}_rewards_a2c', rewards)


class MazeDQNLearning():
    '''DQN Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=0.9, eps_min=0.05, eps_decay=5000, lamb_lasso=0.1, learning_rate=1e-4,
                 update_target=1000, batch_size=20, time_steps=10000, capacity=2000, grid_size=7,
                 stochastic=True, source_name=None, model_dir='maze_project/saved_models/', data_dir='maze_project/data/'):
        '''Initialize the DQN agent with environment and hyperparameters.'''

        self.game_name = game_name
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
        self.source_name = source_name
        self.memory = deque([], maxlen=capacity)
        self.model_dir = model_dir
        self.data_dir = data_dir

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env = gym.make('gym_examples/GridWorld-v0', game=self.game, render_mode='rgb_array')
        no_actions = env.action_space.n

        if self.source_name is not None:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.load_state_dict(torch.load(self.model_dir + self.source_name + '_dqnetwork',
                                             map_location=torch.device(cm.DEVICE)))
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target.load_state_dict(torch.load(self.model_dir + self.source_name + '_dqnetwork',
                                              map_location=torch.device(cm.DEVICE)))
            model.re_init_head()
            target.re_init_head()
            source = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            source.load_state_dict(torch.load(self.model_dir + self.source_name + '_dqnetwork',
                                              map_location=torch.device(cm.DEVICE)))
            source.eval()
            for param in source.parameters():
                param.requires_grad = False

        else:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.initialize_network()
            target.initialize_network()
            source = None

        obs, _ = env.reset()
        state = create_maze_state(obs, self.grid_size)
        rewards = []

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(reward)
            rewards.append(reward)
            cumulative_rewards = np.cumsum(np.asarray(rewards))
            next_state = create_maze_state(next_obs, self.grid_size)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target)

            if (i % self.update_target) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = model.state_dict()[key]
            
            if (i % self.batch_size*10) == 0:
                self.save_model(model)
                self.save_rewards(cumulative_rewards)

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
                       target,
                       source=None):
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

        if source is not None:
            reg_param = hf.biased_lasso_regularization(model, source)


        total_loss += self.lamb_lasso * reg_param
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    def save_model(self, network):
        '''Save models to a specified filename.'''
        torch.save(network.state_dict(), self.model_dir + self.game_name + '_dqn_network')
    
    def save_rewards(self, rewards):
        '''Save the rewards to a specified filename.'''
        np.save(self.data_dir + self.game_name + f'_source_{self.source_name}_lamb_{self.lamb_lasso}_rewards_dqn', rewards)


class MazeDDQNLearning():
    '''Double DQN Agent for different maze tasks, with optional transfer learning.'''

    def __init__(self, game_name, gamma=0.99, epsilon=1, eps_min=0.05, eps_decay=5000, random_phase=1000, lamb_lasso=0.1,
                 learning_rate=1e-4, batch_size=256, target_update=100, tau=1, time_steps=20000, capacity=10000, grid_size=5,
                 stochastic=False, source_name=None, model_dir='maze_project/saved_models/', data_dir='maze_project/data/'):
        '''Initialize the Double DQN agent with environment and hyperparameters.'''

        self.game_name = game_name
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
        self.source_name = source_name
        self.memory = deque([], maxlen=capacity)
        self.model_dir = model_dir
        self.data_dir = data_dir

    def train_agent(self):
        '''Simulate environment and train the agent.'''

        env = gym.make('gym_examples/GridWorld-v0', game=self.game,
                       render_mode='rgb_array', size=self.grid_size)
        no_actions = env.action_space.n
        
        if self.source_name is not None:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.load_state_dict(torch.load(self.model_dir + self.source_name + '_ddqnetwork',
                                             map_location=torch.device(cm.DEVICE)))
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target.load_state_dict(torch.load(self.model_dir + self.source_name + '_ddqnetwork',
                                              map_location=torch.device(cm.DEVICE)))
            model.re_init_head()
            target.re_init_head()
            source = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            source.load_state_dict(torch.load(self.model_dir + self.source_name + '_ddqnetwork',
                                              map_location=torch.device(cm.DEVICE)))
            source.eval()
            for param in source.parameters():
                param.requires_grad = False

        else:
            model = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            target = MazeNetwork(self.grid_size, no_actions).to(cm.DEVICE)
            model.initialize_network()
            target.initialize_network()
            source = None

        obs, _ = env.reset()
        state = create_maze_state(obs, self.grid_size)
        rewards = []

        for i in range(self.time_steps):
            env.render()
            model_output = model(state).to(cm.DEVICE)
            action = self.select_action(model_output, self.epsilon)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(action, reward)
            rewards.append(reward)
            cumulative_rewards = np.cumsum(np.asarray(rewards))

            if i > self.random_phase:
                self.epsilon = max(self.epsilon - i*(self.epsilon-self.eps_min)/self.eps_decay, self.eps_min)

            next_state = create_maze_state(next_obs, self.grid_size)
            self.memory.append(hf.Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.optimize_model(model, target, source)

            if (i % self.target_update) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = self.tau*model.state_dict()[key] + (1-self.tau)*target.state_dict()[key]
            
            if (i % self.batch_size*10) == 0:
                self.save_model(model)
                self.save_rewards(cumulative_rewards)

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

    def optimize_model(self, model, target, source=None):
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

        if source is not None:
            reg_param = hf.biased_lasso_regularization(model, source)


        total_loss += self.lamb_lasso * reg_param
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    def save_model(self, network):
        '''Save models to a specified filename.'''
        torch.save(network.state_dict(), self.model_dir + self.game_name + '_ddqn_network')
    
    def save_rewards(self, rewards):
        '''Save the rewards to a specified filename.'''
        np.save(self.data_dir + self.game_name + f'_source_{self.source_name}_lamb_{self.lamb_lasso}_rewards_ddqn', rewards)


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

    def initialize_network(self, gain=1):
        for layer in [self.layer1, self.layer2, self.layer3, self.layer4, self.layer5]:
            nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            nn.init.zeros_(layer.bias)
        nn.init.orthogonal_(self.layer6.weight, gain=gain)
        nn.init.zeros_(self.layer6.bias)

    def re_init_head(self):
        nn.init.kaiming_uniform_(self.layer5.weight, nonlinearity='relu')
        nn.init.zeros_(self.layer5.bias)
        nn.init.orthogonal_(self.layer6.weight, gain=1)
        nn.init.zeros_(self.layer6.bias)


def create_fc_state_vector_mult_proc(observation, size, n_envs):
    '''Convert Observation output from environmnet into a state variable for regular NN
       with parallelization onto cpu cores.'''

    state_vector = torch.zeros(5, size, size, n_envs).to(cm.DEVICE)
    state_vector[0] =  torch.from_numpy(observation['agent']).to(cm.DEVICE).T
    state_vector[1] =  torch.from_numpy(observation['target']).to(cm.DEVICE).T
    state_vector[2] =  torch.from_numpy(observation['coins']).to(cm.DEVICE).T
    state_vector[3] =  torch.from_numpy(observation['traps']).to(cm.DEVICE).T
    state_vector[4] =  torch.from_numpy(observation['walls']).to(cm.DEVICE).T
    state_vector = torch.reshape(state_vector, (5 * size * size, n_envs))
    return state_vector.T

def create_maze_state(observation, size):
    '''Convert observation value of maze environment to state vector for singular environment'''
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
