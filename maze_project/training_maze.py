# import time
import sys
import os
import random
from collections import deque
import torch
import numpy as np
import gym
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)
from gym_examples.envs.grid_world import GridWorldEnv
# from gym_examples.wrappers import RelativePosition

from . import config_maze as c
import optimization as o
import models as m
import evaluation as e


def train_network(network_model,
                  target_net,
                  no_segments=c.NO_SEGMENTS,
                  game='gym_examples/GridWorld-v0',
                  render_mode=c.RENDER):
    '''Function to train a model given the collected batch data set.'''

    # if c.LOAD_EXPLORATION:
    #     replay_memory = torch.load(c.DATA_DIR + 'exploration_data.pt')
    # else:
    #     replay_memory = deque([], maxlen=c.CAPACITY)

    target_update_counter = 0
    exploration_counter = 0
    eps_decline_counter = 0
    source_network = m.MazeFCNetwork().to(c.DEVICE)
    source_network.init_weights_to_zero()
    acc_reward_array = []

    for t in range(no_segments):

        for param in source_network.parameters():
            param.requires_grad = False

        env = gym.make(game, render_mode=render_mode)
        action_space = env.action_space
        replay_memory = deque([], maxlen=c.CAPACITY)
        for epoch in range(c.EPOCHS):
            obs, _ = env.reset()
            # env.close()

            if c.FCMODEL:
                state = create_fc_state_vector(obs)
            else:
                state = create_conv_state_vector(obs)

            print(f"{epoch} epochs done.")
            accumulated_reward = 0
            mc_explore=False

            for i in range(c.EPISODES):
                env.render()
                # time.sleep(0.1)
                # action = env.action_space.sample()
                epsilon = max(1 - 0.9 * eps_decline_counter/c.EPS_DECLINE, 0.05)
                exploration_counter += 1

                if i > 0 and epsilon > 0.1:
                    mc_explore = True

                else:
                    mc_explore = False

                action = select_action(network_model(state),
                                       epsilon,
                                       mc_explore,
                                       replay_memory,
                                       range(action_space.n))
                next_obs, reward, done, _, _ = env.step(action)

                # if done:
                #     # next_state=None
                #     reward = -1

                # else:
                #     reward = i

                accumulated_reward += reward

                if c.FCMODEL:
                    next_state = create_fc_state_vector(next_obs)
                    replay_memory.append(o.Transition(state, action, next_state, reward))

                else:
                    next_state = create_conv_state_vector(next_obs)
                    replay_memory.append(o.Transition(state[0], action, next_state[0], reward))

                state = next_state
                n_segments = int(len(replay_memory)/c.BATCH_SIZE)

                if len(replay_memory) >= c.BATCH_SIZE:
                    o.optimization_step(network_model, target_net, replay_memory, n_segments, c.GAMMA, c.LEARNING_RATE, c.LAMB, c.BATCH_SIZE, source_network)
                    target_update_counter += 1

                    if target_update_counter == c.UPDATE_TARGET:
                        for key in target_net.state_dict():
                            target_net.state_dict()[key] = c.TAU*network_model.state_dict()[key] + (1 - c.TAU)*target_net.state_dict()[key]

                        target_update_counter = 0

                if exploration_counter >= c.EXPLORATION:

                    if exploration_counter == c.EXPLORATION and c.SAVE_EXPLORATION:
                        torch.save(replay_memory, c.DATA_DIR + 'exploration_data.pt')

                    eps_decline_counter += 1

                if done:
                    print(f"Episode finished after {i+1} timesteps.")
                    break

            print(f'epsilon = {epsilon}')
            print(f'Accumulated a total reward of {accumulated_reward}.')
            acc_reward_array.append(accumulated_reward)

        env.close()
        # source_network = target_net

    e.plot_cumulative_rewards(np.asarray(acc_reward_array), c.EPOCHS)
    return network_model


# def offline_initialization(network_model,
#                            target_model,
#                            replay_memory,
#                            n_epochs=100,
#                            batch_size=50):
#     '''Use saved replay memory data to do offline training in an initialization phase.'''
#     pass


def create_fc_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.zeros(3, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, observation['agent'][0], observation['agent'][1]] = 1   # state_vector[0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[1, observation['target'][0], observation['target'][1]] = 1
    state_vector[2, observation['walls'][0], observation['walls'][1]] = 1
    state_vector = torch.flatten(state_vector)

    return state_vector


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 3, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, 0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[0, 1, observation['target'][0], observation['target'][1]] = 1
    state_vector[0, 2, observation['walls'][0], observation['walls'][1]] = 1

    return state_vector


def select_action(network_output, epsilon, mc_explore, state_history, action_space):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:

        if mc_explore:
            action = monte_carlo_exploration(state_history, action_space)
        else:
            action = np.random.randint(c.GRID_OUTPUT)

    else:
        if torch.max(network_output).isnan():
            print('Nan Value detected')

        # action = int(torch.argmax(network_output))
        prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
        action = np.random.choice(np.arange(4), p=prob_action.cpu().detach().numpy())
    return action


def monte_carlo_exploration(state_history, action_space):
    '''Monte carlo exploration for finding sufficent data for training.'''

    unzipped_history = o.Transition(*zip(*state_history))
    pre_state_history = torch.stack(unzipped_history.state)
    apre_state_history = torch.stack(unzipped_history.next_state)
    action_history = torch.tensor(unzipped_history.action).to(c.DEVICE)
    reward_history = torch.tensor(unzipped_history.reward).to(c.DEVICE)
    current_state = apre_state_history.tolist()[-1]
    total_length = len(action_history.tolist())

    repeated_state_indices=[a for a, i in enumerate(pre_state_history.tolist()) if i==current_state]

    if len(repeated_state_indices) > 0:
        current_state_actions = action_history[repeated_state_indices]
        current_reward_by_action = reward_history[repeated_state_indices]
        action_indices = []
        mean_rewards = []
        ucb_value = []

        for action in action_space:

            action_indices.append([a for a, i in enumerate(current_state_actions.tolist()) if i==action])

            if len(action_indices[-1]) > 0:
                mean_rewards.append(current_reward_by_action[action_indices[-1]].sum()/len(action_indices[-1]))
                ucb_value.append(mean_rewards[-1].to("cpu") + c.MC_EXPLORE_CONST*np.log(total_length)/len(action_indices[-1]))

            else:
                new_action = action
                break

            new_action = int(np.argmax(ucb_value))
            # print(ucb_value)

    else:
        new_action_space = list(action_space)

        if int(action_history[-1])==0:
            del new_action_space[2]

        elif int(action_history[-1])==1:
            del new_action_space[3]

        elif int(action_history[-1])==2:
            del new_action_space[0]

        elif int(action_history[-1])==3:
            del new_action_space[1]

        new_action = random.sample(new_action_space, 1)[0]

    return new_action
