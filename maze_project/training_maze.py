# import time
import sys
import os
import pickle
import random
from collections import deque
import torch
import numpy as np
import gym
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/ls131416/biased_lasso_rl/gym-examples")
import gym_examples
# from gym_examples.envs.grid_world import GridWorldEnv
# from gym_examples.wrappers import RelativePosition

import optimization as o
import models as m
import evaluation as e
from maze_project import config_maze as c


GAME_TASKS = {'pathfinding': 'Paths',
              'coincollecting': 'Coins',
              'trapavoiding': 'Traps',
              'mazenavigating': 'Maze'}


def train_network(no_segments=c.NO_SEGMENTS,
                  render_mode=c.RENDER,
                  conv_net=c.CONVMODEL):
    '''Function to train a model given the collected batch data set.'''

    if c.LOAD_NETWORK:
        if conv_net:
            source_network = m.ConvNetwork().to(c.DEVICE)

        else:
            source_network = m.MazeFCNetwork().to(c.DEVICE)

        source_network.load_state_dict(torch.load(c.MODEL_DIR + 'source_network_segment_0'))
        source_network.eval()
        transfer_learning = True

    else:

        if conv_net:
            source_network = m.ConvNetwork().to(c.DEVICE)

        else:
            source_network = m.MazeFCNetwork().to(c.DEVICE)
            source_network.init_weights_to_zero()
            transfer_learning = False

    time_steps_required_array = []


    for game_name, game in GAME_TASKS.items():

        if c.LOAD_SEGMENT:
        
            with open(c.DATA_DIR+game_name+'_config_dict.pkl', 'rb') as f:
                loaded_dict = pickle.load(f)

            exploration_counter = loaded_dict['exploration_counter']
            eps_decline_counter = loaded_dict['eps_counter']
            target_update_counter = loaded_dict['target_update_counter']
            replay_memory = torch.load(c.DATA_DIR + game_name + '_exploration_data.pt')
            acc_reward_array = torch.load(c.DATA_DIR + game_name + '_rewards.pt')
            loss_array = torch.load(c.DATA_DIR + game_name + '_loss.pt')
            total_loss_array = torch.load(c.DATA_DIR + game_name + '_reg_loss.pt')
            network_model = m.MazeFCNetwork().to(c.DEVICE)
            network_model.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_model'))
            network_model.eval()
            target_net = m.MazeFCNetwork().to(c.DEVICE)
            target_net.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_target_model'))
            target_net.eval()

        else:

            network_model = m.MazeFCNetwork().to(c.DEVICE)
            target_net = m.MazeFCNetwork().to(c.DEVICE)
            acc_reward_array = []
            loss_array = []
            total_loss_array = []
            eps_decline_counter = 0
            exploration_counter = 0
            target_update_counter = 0
            replay_memory = deque([], maxlen=c.CAPACITY)

        for param in target_net.parameters():
            param.requires_grad = False

        for param in source_network.parameters():
            param.requires_grad = False

        for episode in range(c.EPISODES):
            env = gym.make('gym_examples/GridWorld-v0', render_mode=render_mode)
            action_space = env.action_space
            obs, _ = env.reset()
            # env.close()

            if conv_net:
                state = create_conv_state_vector(obs)
            else:
                state = create_fc_state_vector(obs)

            print(f"{episode} episodes done.")
            accumulated_reward = 0
            no_time_steps = 0
            update_q = 0
            mc_explore=False

            for i in range(c.TIME_STEPS):
                env.render()
                # time.sleep(0.1)
                # action = env.action_space.sample()
                epsilon = max(1 - 0.9 * eps_decline_counter/c.EPS_DECLINE, 0.01)
                exploration_counter += 1
                update_q += 1
                term_bool = 1

                if i > 0 and epsilon > 0.01:
                    mc_explore = True

                else:
                    mc_explore = False

                action = select_action(network_model(state),
                                       epsilon,
                                       mc_explore,
                                       replay_memory,
                                       range(action_space.n))
                next_obs, reward, done, _, _ = env.step(action)
                no_time_steps += 1

                if done:
                    term_bool = 0

                accumulated_reward += reward

                if conv_net:
                    next_state = create_conv_state_vector(next_obs)
                    replay_memory.append(o.Transition(state[0],
                                                      action,
                                                      next_state[0],
                                                      reward,
                                                      term_bool))

                else:
                    next_state = create_fc_state_vector(next_obs)
                    # ADD ORDER FUNCTION TO ORDER BY PRIORITY OF REPLAY
                    replay_memory.append(o.Transition(state, action, next_state, reward, term_bool))

                state = next_state
                n_segments = int(len(replay_memory)/c.BATCH_SIZE)

                if len(replay_memory) >= c.BATCH_SIZE and len(replay_memory) >= c.EXPLORATION and update_q >= c.UPDATE_Q:
                    loss_value, total_loss = o.optimization_step(network_model,
                                                                 target_net,
                                                                 replay_memory,
                                                                 c.GAMMA,
                                                                 c.LEARNING_RATE,
                                                                 c.MOMENTUM,
                                                                 c.LAMB_LASSO,
                                                                 c.LAMB_RIDGE,
                                                                 c.BATCH_SIZE,
                                                                 source_network,
                                                                 transfer_learning)
                    target_update_counter += 1
                    update_q = 0
                    loss_array.append(loss_value)
                    total_loss_array.append(total_loss)

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
            time_steps_required_array.append(no_time_steps)

        if c.SAVE_NETWORK:
            torch.save(network_model.state_dict(), c.MODEL_DIR +'source_network_segment_' + game_name)

        env.close()
        for key in network_model.state_dict():
            source_network.state_dict()[key] = network_model.state_dict()[key]

        e.plot_nn_weights(network_model, t, c.PLOT_DIR)
        network_model = m.MazeFCNetwork().to(c.DEVICE)

    if c.SAVE_SEGMENT:

        config_dict = {
                        "eps_counter": eps_decline_counter,
                        "exploration_counter": exploration_counter,
                        "target_update_counter": target_update_counter}

        with open(c.DATA_DIR + game_name+'_config_dict.pkl', 'wb') as f:
            pickle.dump(config_dict, f)

        torch.save(network_model.state_dict(), c.MODEL_DIR + game_name + '_model')
        torch.save(target_net.state_dict(), c.MODEL_DIR + game_name + '_target_model')
        torch.save(replay_memory, c.DATA_DIR + game_name + '_exploration_data.pt')
        torch.save(acc_reward_array, c.DATA_DIR + game_name + '_rewards.pt')
        torch.save(loss_array, c.DATA_DIR + game_name + '_loss.pt')
        torch.save(total_loss_array, c.DATA_DIR + game_name + '_reg_loss.pt')


    e.plot_cumulative_rewards_per_segment(np.cumsum(np.asarray(acc_reward_array)),
                                          c.EPISODES*no_segments,
                                          c.EPISODES,
                                          c.NO_SEGMENTS,
                                          plot_dir=c.PLOT_DIR)
    e.plot_time_steps_in_maze(time_steps_required_array,
                              c.EPISODES*no_segments,
                              c.EPISODES,
                              c.NO_SEGMENTS,
                              plot_dir=c.PLOT_DIR)
    e.plot_loss_function(loss_array,
                         len(loss_array),
                         plot_dir=c.PLOT_DIR)
    e.plot_loss_function(total_loss_array,
                         len(total_loss_array),
                         y_label='Regularized Loss',
                         plot_suffix='Total_loss_evolution',
                         plot_dir=c.PLOT_DIR)

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

    state_vector = torch.zeros(5, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[1, observation['target'][0], observation['target'][1]] = 1

    for coin in observation['coins']:
        state_vector[2, coin[0], coin[1]] = 1

    for trap in observation['traps']:
        state_vector[2, trap[0], trap[1]] = 1

    for wall in observation['walls']:
        state_vector[2, wall[0], wall[1]] = 1

    state_vector = torch.flatten(state_vector)

    return state_vector


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 3, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, 0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[0, 1, observation['target'][0], observation['target'][1]] = 1
    state_vector[0, 2, observation['coins'][0], observation['coins'][1]] = 1
    state_vector[0, 3, observation['traps'][0], observation['traps'][1]] = 1
    state_vector[0, 4, observation['walls'][0], observation['walls'][1]] = 1

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
        # action = np.random.choice(np.arange(4), p=prob_action.cpu().detach().numpy())
        action = np.argmax(torch.flatten(network_output).cpu().detach().numpy())
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
                ucb_value.append(np.inf)
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
