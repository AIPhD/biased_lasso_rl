import time
import torch
from torch import nn
from gym_examples.envs.grid_world import GridWorldEnv
from gym_examples.wrappers import RelativePosition
import gym
import matplotlib.pyplot as plt
import numpy as np
import config as c
import optimization as o


def train_network(network_model, render_mode='rgb_array'):
    '''Function to train a model given the collected batch data set.'''

    # batch_size = len(batch_data)
    env = gym.make('gym_examples/GridWorld-v0', render_mode=render_mode)
    # env = gym.make('Pong-v0')
    wrapped_env = RelativePosition(env)
    for epoch in range(c.EPOCHS):
        env.reset()
        env.close()
        for i in range(30):
            env.render()
            time.sleep(0.1)
            action = env.action_space.sample()
            observation, reward, done, info, dis = env.step(action)
            state_vector = create_state_vector(observation)
            y_batch = reward + c.GAMMA * torch.max(nn.Softmax(network_model.forward(state_vector)))
            y_pred = nn.functional.softmax(network_model.forward(state_vector))
            loss = nn.MSELoss(y_pred, y_batch)
            loss.backward()
            o.optimizer.step()
            if done:
                print("Episode finished after {} timesteps".format(i+1))
            break

def create_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for NN.'''
    state_vector = torch.zeros(5,5,2)
    state_vector[observation['agent'][0], observation['agent'][1], 0] = 1
    state_vector[observation['target'][0], observation['target'][1], 1] = 1
    return torch.flatten(state_vector)
