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

    env = gym.make('gym_examples/GridWorld-v0', render_mode=render_mode)
    # env = gym.make('Pong-v0')
    # wrapped_env = RelativePosition(env)
    if c.LOAD_EXPLORATION:
        replay_memory = torch.load(c.DATA_DIR + 'exploration_data.pt')
    else:
        replay_memory = deque([], maxlen=c.CAPACITY)
    target_update_counter = 0
    exploration_counter = 0
    eps_decline_counter = 0
    for epoch in range(c.EPOCHS):
        obs, info = env.reset()
        # env.close()

        if c.FCMODEL:
            state = create_fc_state_vector(obs)
        else:
            state = create_conv_state_vector(obs)

        print(f"{epoch} epochs done.")
        accumulated_reward = 0
        for i in range(c.EPISODES):
            env.render()
            # time.sleep(0.1)
            # action = env.action_space.sample()
            epsilon = max(1 - 0.9 * eps_decline_counter/c.EPS_DECLINE, 0.1)
            exploration_counter += 1
            action = select_action(network_model(state), epsilon)
            next_obs, reward, done, info, dis = env.step(action)
            accumulated_reward += reward

            if c.FCMODEL:
                state = create_fc_state_vector(obs)
                next_state = create_fc_state_vector(next_obs)
                replay_memory.append(o.Transition(state, action, next_state, reward))
            else:
                state = create_conv_state_vector(obs)
                next_state = create_conv_state_vector(next_obs)
                replay_memory.append(o.Transition(state[0], action, next_state[0], reward))

            state = next_state
            n_segments = int(len(replay_memory)/c.BATCH_SIZE)

            if exploration_counter >= c.EXPLORATION:

                if exploration_counter == c.EXPLORATION and c.SAVE_EXPLORATION:
                    torch.save(replay_memory, c.DATA_DIR + 'exploration_data.pt')

                eps_decline_counter += 1
                target_update_counter += 1
                o.optimization_step(network_model, target_net, replay_memory, n_segments)

                if target_update_counter == c.UPDATE_TARGET:
                    for key in target_net.state_dict():
                        target_net.state_dict()[key] = network_model.state_dict()[key]

                    target_update_counter = 0

            if done:
                print(f"Episode finished after {i+1} timesteps.")
                break

        print(f'epsilon = {epsilon}')
        print(f'Accumulated a total reward of {accumulated_reward}.')
    return network_model


def offline_initialization(network_model, target_model, replay_memory, n_epochs=100, batch_size=50):
    '''Use saved replay memory data to do offline training in an initialization phase.'''
    pass


def create_fc_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.zeros(2, 8, 8).to(c.DEVICE)
    state_vector[0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[1, observation['target'][0], observation['target'][1]] = 1
    state_vector = torch.flatten(state_vector)

    return state_vector


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 2, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, 0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[0, 1, observation['target'][0], observation['target'][1]] = 1

    return state_vector


def select_action(network_output, epsilon):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:
        action = np.random.randint(4)
    else:
        action = int(torch.argmax(network_output))
    #prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
    #action = np.random.choice(np.arange(4), p=prob_action.cpu().detach().numpy())
    return action


def monte_carlo_exploration(state_history):
    pass
