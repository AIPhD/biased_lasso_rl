import time
from collections import deque
import torch
from gym_examples.envs.grid_world import GridWorldEnv
from gym_examples.wrappers import RelativePosition
import gym
import matplotlib.pyplot as plt
import numpy as np
import config as c
import optimization as o
import models as m


def train_network(network_model, render_mode=c.RENDER):
    '''Function to train a model given the collected batch data set.'''

    # batch_size = len(batch_data)
    env = gym.make('gym_examples/GridWorld-v0', render_mode=render_mode)
    # env = gym.make('Pong-v0')
    # wrapped_env = RelativePosition(env)
    replay_memory = deque([], maxlen=c.CAPACITY)
    model_optimizer = o.NetworkOptimizer(network_model)
    for epoch in range(c.EPOCHS):
        obs, info = env.reset()
        # env.close()
        # obs, info = env.reset()
        state = create_state_vector(obs)
        print(epoch)
        for i in range(100):
            env.render()
            time.sleep(0.1)
            # action = env.action_space.sample()
            action = select_action(network_model(state))
            next_obs, reward, done, info, dis = env.step(action)
            state = create_state_vector(obs)
            next_state = create_state_vector(next_obs)
            replay_memory.append(o.Transition(state, action, next_state, reward))
            state = next_state
            if i > 0:
                model_optimizer.optimization_step(replay_memory)
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
    return network_model

def create_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for NN.'''
    state_vector = torch.zeros(5,5,2)
    state_vector[observation['agent'][0], observation['agent'][1], 0] = 1
    state_vector[observation['target'][0], observation['target'][1], 1] = 1
    return torch.flatten(state_vector)


def select_action(network_output):
    '''Calculate, which action to take, with best estimated chance.'''
    prob_action = torch.nn.functional.softmax(network_output, dim=0)
    print(prob_action)
    action = np.random.choice(np.arange(4), p=prob_action.detach().numpy())
    return action
