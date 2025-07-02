import random
import sys
import pickle
from collections import deque
import numpy as np
import torch
from torch import optim
import gym
import optimization as o
import models as m
import evaluation as e
from atari_project import config_atari as c


ATARI_GAMES = {
               'Alien': 'ALE/Alien-v5',
               'Centipede': 'ALE/Centipede-v5',
               'Jamesbond': 'ALE/Jamesbond-v5',
               'Pong': 'ALE/Pong-v5',
               'Seaquest': 'ALE/Seaquest-v5',
               'Riverraid': 'ALE/Riverraid-v5'
               }

def train_dqn_network(render_mode=c.RENDER,
                      games=ATARI_GAMES,
                      transfer_learning=False):
    '''Function to run and train a dqn model. Data is collected given an epsilon-greedy policy'''

    # if c.LOAD_EXPLORATION:
    #     replay_memory = torch.load(c.DATA_DIR + 'exploration_data.pt')
    # else:
    #     replay_memory = deque([], maxlen=c.CAPACITY)


    if c.LOAD_NETWORK:

        source_network = m.AtariNetwork(action_space.n).to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR))
        source_network.eval()
        transfer_learning = True

    else:
        source_network = m.AtariNetwork(action_space.n).to(c.DEVICE)
        transfer_learning = False

    for param in source_network.parameters():
            param.requires_grad = False

    for game_name, game in games.items():

        env = gym.make(game, render_mode=render_mode, frameskip=1)
        action_space = env.action_space

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
            network_model = m.AtariNetwork(action_space.n).to(c.DEVICE)
            network_model.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_model'))
            network_model.eval()
            target_net = m.AtariNetwork(action_space.n).to(c.DEVICE)
            target_net.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_target_model'))
            target_net.eval()

        else:

            network_model = m.AtariNetwork(action_space.n).to(c.DEVICE)
            target_net = m.AtariNetwork(action_space.n).to(c.DEVICE)
            acc_reward_array = []
            loss_array = []
            total_loss_array = []
            eps_decline_counter = 0
            exploration_counter = 0
            target_update_counter = 0
            replay_memory = deque([], maxlen=c.CAPACITY)

        for param in target_net.parameters():
            param.requires_grad = False

        env = gym.wrappers.AtariPreprocessing(env,
                                              noop_max=30,
                                              frame_skip=c.FRAME_SKIP,
                                              screen_size=84,
                                              terminal_on_life_loss=False,
                                              grayscale_obs=True,
                                              grayscale_newaxis=False,
                                              scale_obs=False)
        env = gym.wrappers.FrameStack(env, 4)

        for episode in range(c.EPISODES):
            obs, _ = env.reset()
            # env.close()
            state = create_conv_state_vector(obs)
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
                                       action_space.n,
                                       epsilon)
                next_obs, reward, done, _, _ = env.step(action)

                if done:
                    term_bool = 0

                accumulated_reward += reward
                next_state = create_conv_state_vector(next_obs)
                replay_memory.append(o.Transition(state[0],
                                     action,
                                     next_state[0],
                                     reward,
                                     term_bool))
                state = next_state
                update_q += 1

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
                    loss_array.append(loss_value)
                    total_loss_array.append(total_loss)
                    update_q = 0

                    if target_update_counter == c.UPDATE_TARGET:
                        for key in target_net.state_dict():
                            target_net.state_dict()[key] = c.TAU*network_model.state_dict()[key] + (1 - c.TAU)*target_net.state_dict()[key]

                        target_update_counter = 0

                if exploration_counter >= c.EXPLORATION:

                    eps_decline_counter += 1

                if done:
                    print(f"Episode finished after {i+1} timesteps.")
                    break

            print(f"{episode + 1} episode(s) done.")
            print(f'epsilon = {epsilon}')
            print(f'Accumulated a total reward of {accumulated_reward}.')
            acc_reward_array.append(accumulated_reward)

            if (episode + 1) % c.SAVE_PERIOD == 0 and c.SAVE_SEGMENT:

                print('Save Plots and Data')

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

                e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
                                                      len(np.asarray(acc_reward_array))*len(ATARI_GAMES),
                                                      len(np.asarray(acc_reward_array)),
                                                      len(ATARI_GAMES),
                                                      plot_suffix=game_name+'_accumulated_reward',
                                                      plot_dir=c.PLOT_DIR)
                e.plot_loss_function(loss_array,
                                     len(loss_array),
                                     plot_suffix=game_name+'_loss_evolution',
                                     plot_dir=c.PLOT_DIR)
                e.plot_loss_function(total_loss_array,
                                     len(total_loss_array),
                                     y_label='Regularized Loss',
                                     plot_suffix=game_name+'_total_loss_evolution',
                                     plot_dir=c.PLOT_DIR)

        env.close()

    # if c.SAVE_SEGMENT:

    #     config_dict = {
    #                     "eps_counter": eps_decline_counter,
    #                     "exploration_counter": exploration_counter,
    #                     "target_update_counter": target_update_counter}

    #     with open(c.DATA_DIR + game_name+'_config_dict.pkl', 'wb') as f:
    #         pickle.dump(config_dict, f)

    #     torch.save(network_model.state_dict(), c.MODEL_DIR + game_name + '_model')
    #     torch.save(target_net.state_dict(), c.MODEL_DIR + game_name + '_target_model')
    #     torch.save(replay_memory, c.DATA_DIR + game_name + '_exploration_data.pt')
    #     torch.save(acc_reward_array, c.DATA_DIR + game_name + '_rewards.pt')
    #     torch.save(loss_array, c.DATA_DIR + game_name + '_loss.pt')
    #     torch.save(total_loss_array, c.DATA_DIR + game_name + '_reg_loss.pt')


    # e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
    #                                       len(np.asarray(acc_reward_array))*len(ATARI_GAMES),
    #                                       len(np.asarray(acc_reward_array)),
    #                                       len(ATARI_GAMES),
    #                                       plot_suffix=game_name+'_accumulated_reward',
    #                                       plot_dir=c.PLOT_DIR)
    # e.plot_loss_function(loss_array,
    #                      len(loss_array),
    #                      plot_suffix=game_name+'_loss_evolution',
    #                      plot_dir=c.PLOT_DIR)
    # e.plot_loss_function(total_loss_array,
    #                      len(total_loss_array),
    #                      y_label='Regularized Loss',
    #                      plot_suffix=game_name+'_total_loss_evolution',
    #                      plot_dir=c.PLOT_DIR)

    return network_model


def train_actor_critic(game_name):
    '''Train policy based on a regular actor-critic framework.'''

    if c.LOAD_NETWORK:

        source_network = m.AtariValueNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR))
        source_network.eval()
        transfer_learning = True

    else:
        source_network = m.AtariValueNetwork().to(c.DEVICE)
        transfer_learning = False

    for param in source_network.parameters():
            param.requires_grad = False

    if c.LOAD_SEGMENT:

        acc_reward_array = torch.load(c.DATA_DIR + game_name + '_rewards_a3c.pt')
        loss_array = torch.load(c.DATA_DIR + game_name + '_loss_a3c.pt')
        total_loss_array = torch.load(c.DATA_DIR + game_name + '_reg_loss_a3c.pt')
        network_policy = m.AtariPolicyNetwork().to(c.DEVICE)
        network_policy.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_policy'))
        network_policy.eval()
        network_value_function = m.AtariValueNetwork().to(c.DEVICE)
        network_value_function.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_value'))
        network_value_function.eval()

    else:

        network_policy = m.AtariPolicyNetwork().to(c.DEVICE)
        network_value_function = m.AtariValueNetwork().to(c.DEVICE)
        acc_reward_array = []
        loss_array = []
        total_loss_array = []

    env = gym.make(ATARI_GAMES[game_name], render_mode=c.RENDER, frameskip=1)
    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=30,
                                          frame_skip=c.FRAME_SKIP,
                                          screen_size=84,
                                          terminal_on_life_loss=False,
                                          grayscale_obs=True,
                                          grayscale_newaxis=False,
                                          scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    action_space = env.action_space

    for episode in range(c.EPISODES):

        obs, _ = env.reset()
        # env.close()
        state = create_conv_state_vector(obs)
        accumulated_reward = 0

        for i in range(c.TIME_STEPS):
            env.render()
            # time.sleep(0.1)
            # action = env.action_space.sample()
            term_bool = 1
            action = select_action(network_policy(state),
                                   1)

            next_obs, reward, done, _, _ = env.step(action)

            if done:
                term_bool = 0

            accumulated_reward += reward
            next_state = create_conv_state_vector(next_obs)
            loss_value, total_loss = o.actor_critic_optimization(network_policy,
                                                                 network_value_function,
                                                                 reward,
                                                                 action,
                                                                 state,
                                                                 next_state,
                                                                 term_bool,
                                                                 c.GAMMA,
                                                                 learning_rate=c.LEARNING_RATE)
            loss_array.append(loss_value)
            total_loss_array.append(total_loss)
            state = next_state


            if done:
                print(f"Episode finished after {i+1} timesteps.")
                break

        print(f"{episode + 1} episode(s) done.")
        print(f'Accumulated a total reward of {accumulated_reward}.')
        acc_reward_array.append(accumulated_reward)

        if (episode + 1) % c.SAVE_PERIOD == 0 and c.SAVE_SEGMENT:

            print('Save Plots and Data')
            torch.save(network_policy.state_dict(), c.MODEL_DIR + game_name + '_policy')
            torch.save(network_value_function.state_dict(), c.MODEL_DIR + game_name + '_value')
            torch.save(acc_reward_array, c.DATA_DIR + game_name + '_rewards_a3c.pt')
            torch.save(loss_array, c.DATA_DIR + game_name + '_loss_a3c.pt')
            torch.save(total_loss_array, c.DATA_DIR + game_name + '_reg_loss_a3c.pt')

            e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
                                                  len(np.asarray(acc_reward_array)),
                                                  len(np.asarray(acc_reward_array)),
                                                  1,
                                                  plot_suffix=game_name+'_accumulated_reward_a3c',
                                                  plot_dir=c.PLOT_DIR)
            e.plot_loss_function(loss_array,
                                 len(loss_array),
                                 plot_suffix=game_name+'_loss_evolution_a3c',
                                 plot_dir=c.PLOT_DIR)
            # e.plot_loss_function(total_loss_array,
            #                      len(total_loss_array),
            #                      y_label='Regularized Loss',
            #                      plot_suffix=game_name+'_total_loss_evolution_a3c',
            #                      plot_dir=c.PLOT_DIR)

    env.close()


def train_a2c_network(game_name):
    '''Train policy based on the advantage a2c framework.'''

    if c.LOAD_NETWORK:

        source_network = m.AtariValueNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR))
        source_network.eval()
        transfer_learning = True

    else:
        source_network = m.AtariValueNetwork().to(c.DEVICE)
        transfer_learning = False

    for param in source_network.parameters():
            param.requires_grad = False

    env = gym.make(ATARI_GAMES[game_name], render_mode=c.RENDER, frameskip=1)
    action_space = env.action_space

    if c.LOAD_SEGMENT:

        acc_reward_array = torch.load(c.DATA_DIR + game_name + '_rewards_a3c.pt')
        loss_array = torch.load(c.DATA_DIR + game_name + '_loss_a3c.pt')
        total_loss_array = torch.load(c.DATA_DIR + game_name + '_reg_loss_a3c.pt')
        network_policy = m.AtariPolicyNetwork(action_space.n).to(c.DEVICE)
        network_policy.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_policy'))
        network_policy.eval()
        network_value_function = m.AtariValueNetwork().to(c.DEVICE)
        network_value_function.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_value'))
        network_value_function.eval()

    else:

        network_policy = m.AtariPolicyNetwork(action_space.n).to(c.DEVICE)
        network_value_function = m.AtariValueNetwork().to(c.DEVICE)
        acc_reward_array = []
        loss_array = []
        total_loss_array = []

    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=30,
                                          frame_skip=c.FRAME_SKIP,
                                          screen_size=84,
                                          terminal_on_life_loss=False,
                                          grayscale_obs=True,
                                          grayscale_newaxis=False,
                                          scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)

    for episode in range(c.EPISODES):

        obs, _ = env.reset()
        # env.close()
        state = create_conv_state_vector(obs)
        accumulated_reward = 0
        replay_memory = deque([], maxlen=c.CAPACITY)

        for i in range(c.TIME_STEPS):
            env.render()
            # time.sleep(0.1)
            # action = env.action_space.sample()
            term_bool = 1
            action = select_action(network_policy(state),
                                   action_space.n,
                                   1)

            next_obs, reward, done, _, _ = env.step(action)

            if done:
                term_bool = 0

            big_r = term_bool*network_value_function(state)

            accumulated_reward += reward
            next_state = create_conv_state_vector(next_obs)
            replay_memory.append(o.Transition(state[0],
                                              action,
                                              next_state[0],
                                              reward,
                                              term_bool))
            state = next_state

            if done:
                print(f"Episode finished after {i+1} timesteps.")
                break

        loss_value, total_loss = o.a2c_optimization(network_policy,
                                                    network_value_function,
                                                    replay_memory,
                                                    big_r,
                                                    c.GAMMA,
                                                    learning_rate=c.LEARNING_RATE)
        loss_array.append(loss_value)
        total_loss_array.append(total_loss)
        print(f"{episode + 1} episode(s) done.")
        print(f'Accumulated a total reward of {accumulated_reward}.')
        acc_reward_array.append(accumulated_reward)

        if (episode + 1) % c.SAVE_PERIOD == 0 and c.SAVE_SEGMENT:

            print('Save Plots and Data')
            torch.save(network_policy.state_dict(), c.MODEL_DIR + game_name + '_policy')
            torch.save(network_value_function.state_dict(), c.MODEL_DIR + game_name + '_value')
            torch.save(acc_reward_array, c.DATA_DIR + game_name + '_rewards_a3c.pt')
            torch.save(loss_array, c.DATA_DIR + game_name + '_loss_a3c.pt')
            torch.save(total_loss_array, c.DATA_DIR + game_name + '_reg_loss_a3c.pt')

            e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
                                                  len(np.asarray(acc_reward_array)),
                                                  len(np.asarray(acc_reward_array)),
                                                  1,
                                                  plot_suffix=game_name+'_accumulated_reward_a3c',
                                                  plot_dir=c.PLOT_DIR)
            e.plot_loss_function(loss_array,
                                 len(loss_array),
                                 plot_suffix=game_name+'_loss_evolution_a3c',
                                 plot_dir=c.PLOT_DIR)
            # e.plot_loss_function(total_loss_array,
            #                      len(total_loss_array),
            #                      y_label='Regularized Loss',
            #                      plot_suffix=game_name+'_total_loss_evolution_a3c',
            #                      plot_dir=c.PLOT_DIR)

    env.close()


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 4, 84, 84).to(c.DEVICE)
    state_vector[0] = torch.from_numpy(np.reshape(observation, (4, 84, 84)))
    return state_vector


def select_action(network_output, number_actions, epsilon, stochastic=True):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:
        if torch.max(network_output).isnan():

            sys.exit("Nan Value detected, Network diverged.")

        if stochastic:

            prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
            action = np.random.choice(np.arange(number_actions), p=prob_action.cpu().detach().numpy())

        else:

            action = np.argmax(torch.flatten(network_output).cpu().detach().numpy())

    else:
        action = np.random.randint(number_actions)

    return action
