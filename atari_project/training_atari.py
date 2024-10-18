import random
from collections import deque
import numpy as np
import torch
import gym
import optimization as o
import models as m
import evaluation as e
from atari_project import config_atari as c


ATARI_GAMES = [
               'ALE/Alien-v5',
            #    'ALE/Centipede-v5',
            #    'ALE/Jamesbond-v5'
               ]

def train_network(network_model,
                  target_net,
                  render_mode=c.RENDER,
                  games=ATARI_GAMES,
                  transfer_learning=False):
    '''Function to train a model given the collected batch data set.'''

    # if c.LOAD_EXPLORATION:
    #     replay_memory = torch.load(c.DATA_DIR + 'exploration_data.pt')
    # else:
    #     replay_memory = deque([], maxlen=c.CAPACITY)

    target_update_counter = 0
    exploration_counter = 0
    eps_decline_counter = 0

    if c.LOAD_NETWORK:

        source_network = m.AtariNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR))
        source_network.eval()

    else:
        source_network = m.AtariNetwork().to(c.DEVICE)

    acc_reward_array = []
    loss_array = []

    for game in games:

        for param in source_network.parameters():
            param.requires_grad = False

        env = gym.make(game, render_mode=render_mode)
        action_space = env.action_space
        replay_memory = deque([], maxlen=c.CAPACITY)
        for episode in range(c.EPISODES):
            obs, _ = env.reset()
            # env.close()
            state = create_conv_state_vector(obs)
            print(f"{episode} episodes done.")
            accumulated_reward = 0
            update_q = 0

            for i in range(c.TIME_STEPS):
                env.render()
                # time.sleep(0.1)
                # action = env.action_space.sample()
                epsilon = max(1 - 0.9 * eps_decline_counter/c.EPS_DECLINE, 0.05)
                exploration_counter += 1
                term_bool = 1
                action = select_action(network_model(state),
                                       epsilon,
                                       replay_memory,
                                       range(action_space.n))
                next_obs, reward, done, _, _ = env.step(action)

                if done or i==c.TIME_STEPS - 1:
                    term_bool = 0

                accumulated_reward += reward
                next_state = create_conv_state_vector(next_obs)
                replay_memory.append(o.Transition(state[0],
                                     action,
                                     next_state[0],
                                     reward,
                                     term_bool))
                state = next_state

                if len(replay_memory) >= c.BATCH_SIZE and len(replay_memory) >= c.EXPLORATION and update_q >= c.UPDATE_Q:
                    loss_value = o.optimization_step(network_model,
                                                     target_net,
                                                     replay_memory,
                                                     c.GAMMA,
                                                     c.LEARNING_RATE,
                                                     c.LAMB_LASSO,
                                                     c.LAMB_RIDGE,
                                                     c.BATCH_SIZE,
                                                     source_network,
                                                     transfer_learning)
                    target_update_counter += 1
                    update_q += 1
                    loss_array.append(loss_value)

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

        if c.SAVE_NETWORK:
            torch.save(network_model.state_dict(), c.MODEL_DIR +'source_network_segment_' + game)

        env.close()

        for key in target_net.state_dict():
            source_network.state_dict()[key] = target_net.state_dict()[key]


    e.plot_cumulative_rewards_per_segment(np.cumsum(np.asarray(acc_reward_array)),
                                          c.EPISODES*no_segments,
                                          c.EPISODES,
                                          c.NO_SEGMENTS,
                                          plot_dir=c.PLOT_DIR)
    e.plot_loss_function(loss_array,
                         len(loss_array),
                         plot_dir=c.PLOT_DIR)

    return network_model


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 3, 96, 96).to(c.DEVICE)
    state_vector[0] = torch.from_numpy(np.reshape(observation, (3, 96,96)))
    return state_vector


def select_action(network_output, epsilon, state_history, action_space):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:

        action = np.random.randint(18)

    else:
        if torch.max(network_output).isnan():
            print('Nan Value detected')

        # action = int(torch.argmax(network_output))
        # prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
        # action = np.random.choice(np.arange(4), p=prob_action.cpu().detach().numpy())
        action = np.argmax(torch.flatten(network_output).cpu().detach().numpy())
    return action
