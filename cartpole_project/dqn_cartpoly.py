import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from collections import namedtuple, deque
import gymnasium as gym
import random
import json

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

def load_config(file):
    with open(file, 'r') as f:
        cfg = json.load(f)
    return cfg


class NeuralNet(nn.Module):
    
    def __init__(self, inp_dim, out_dim):

        super().__init__()
        self.stacked_layers = nn.Sequential(nn.Linear(inp_dim, 16),
                                            nn.ReLU(),
                                            nn.Linear(16, 32),
                                            nn.ReLU(),
                                            nn.Linear(32, out_dim))
        
    def forward(self, inp):
        out = self.stacked_layers(inp)
        return out


class RL_Simulation():

    def __init__(self, learning_rate=1e-4, time_steps=1000, max_capacity=1000, epsilon=0.9, batch_size=32, gamma=0.9, update_target=100):
        self.learning_rate = learning_rate
        self.time_steps = time_steps
        self.memory = deque([], maxlen=max_capacity)
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.gamma = 0.9
        self.update_target = update_target

    def dqn_step(self, model, target):
        optimizer = optim.Adam(model.parameters(),
                               lr=self.learning_rate)
        loss = nn.MSELoss()
        sample = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*sample))
        states = torch.stack(batch.state)
        next_states =torch.stack(batch.next_state)
        rewards = torch.tensor(batch.reward)
        actions = torch.tensor(batch.action)
        terms = torch.tensor(batch.terminal)
        predicted = model(states).gather(1, actions[:, None]).flatten()
        target = rewards + self.gamma * (torch.ones(self.batch_size) - terms) * torch.max(target(next_states), 1).values
        final_loss = loss(target.detach(), predicted)
        final_loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    def training_net(self):

        env = gym.make('CartPole-v1')
        action_space = env.action_space
        obs, info = env.reset()
        state = torch.from_numpy(obs)
        model = NeuralNet(len(obs), action_space.n)
        target = NeuralNet(len(state), action_space.n)

        for i in range(self.time_steps):

            model_out = model(state)
            action = self.select_action(model_out)
            next_obs, reward, term, trunc, _ = env.step(action)
            print(term)
            next_state = torch.from_numpy(next_obs)
            self.memory.append(Transition(state, action, next_state, reward, int(term)))
            state = next_state

            if i > self.batch_size:
                self.dqn_step(model, target)
            
            if term or trunc:
                obs, info = env.reset()
                state = torch.from_numpy(obs)

            if (i % self.update_target) == 0:
                for key in model.state_dict():
                    target.state_dict()[key] = model.state_dict()[key]
        
        env.close()

    
    def select_action(self, model_out):

        threshold = np.random.sample()

        if threshold > self.epsilon:
            action = np.random.randint(len(model_out))

        else:
            action = np.argmax(model_out.cpu().detach().numpy())

        return action
  


cfg = load_config('config.json')
test_sim = RL_Simulation(learning_rate=cfg['learning_rate'],
                         time_steps=cfg['time_steps'],
                         max_capacity=cfg['max_capacity'],
                         epsilon=cfg['epsilon'],
                         batch_size=cfg['batch_size'],
                         gamma=cfg['gamma'],
                         update_target=cfg['update_target'])
test_sim.training_net()