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


def train_network(network_model, target_net, render_mode=c.RENDER):
    '''Function to train a model given the collected batch data set.'''

    # batch_size = len(batch_data)
    env = gym.make('gym_examples/GridWorld-v0', render_mode=render_mode)
    # env = gym.make('Pong-v0')
    # wrapped_env = RelativePosition(env)
    replay_memory = deque([], maxlen=c.CAPACITY)
    model_optimizer = o.NetworkOptimizer(network_model)
    target_update_counter = 0
    for epoch in range(c.EPOCHS):
        obs, info = env.reset()
        # env.close()
        # obs, info = env.reset()
        state = create_state_vector(obs)
        print(epoch)
        accumulated_reward = 0
        for i in range(c.EPISODES):
            target_update_counter += 1
            env.render()
            time.sleep(0.1)
            # action = env.action_space.sample()
            action = select_action(network_model(state))
            next_obs, reward, done, info, dis = env.step(action)
            print(reward)
            accumulated_reward += reward
            state = create_state_vector(obs)
            next_state = create_state_vector(next_obs)
            replay_memory.append(o.Transition(state, action, next_state, reward))
            state = next_state
            if done:
                print("Episode finished after {} timesteps".format(i+1))
                break
            n_segments = int(len(replay_memory)/c.BATCH_SIZE)
            if len(replay_memory) >= c.BATCH_SIZE:
                model_optimizer.optimization_step(target_net, replay_memory, n_segments)
                if target_update_counter == c.UPDATE_TARGET:
                    for key in target_net.state_dict():
                        target_net.state_dict()[key] = network_model.state_dict()[key]
                    target_update_counter = 0
        print(accumulated_reward)
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
    # print(prob_action)
    action = np.random.choice(np.arange(4), p=prob_action.detach().numpy())
    return action
