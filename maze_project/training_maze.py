# import time
import sys
import os
import pickle
import random
from collections import deque
import torch
import numpy as np
import multiprocessing as mp
from stable_baselines3.common.vec_env import SubprocVecEnv
import gymnasium as gym
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append("/home/ls131416/biased_lasso_rl/gym-examples")
import gym_examples
# from gym_examples.envs.grid_world import GridWorldEnv
# from gym_examples.wrappers import RelativePosition

import optimization as o
import models as m
import evaluation as e
from maze_project import config_maze as c


GAME_TASKS = {'pathfinding': [1, 0, 0, 0],
              'coincollecting': [1, 1, 0, 0],
              'onlycoincollecting': [0, 1, 0, 0],
              'trapavoiding': [1, 0, 1, 0],
              'onlytraps': [0, 0, 1, 0],
              'mazenavigating': [1, 0, 0, 1],
              'coinsandtraps': [1, 1, 1, 0],
              'onlycoinsandtraps': [0, 1, 1, 0],
              'coinsinmaze': [1, 1, 0, 1],
              'onlycoinsinmaze': [0, 1, 0, 1],
              'trapsinmaze': [1, 0, 1, 1],
              'completetask': [1, 1, 1, 1]
              }


def make_env(seed, game):
    def _init():
        env = gym.make('gym_examples/GridWorld-v0',
                       game=game,
                       render_mode=c.RENDER)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=250)
        env.reset(seed=seed)
        return env
    return _init    


def train_dqn_network(no_segments=c.NO_SEGMENTS,
                  render_mode=c.RENDER,
                  conv_net=c.CONVMODEL,
                  source_name=None):
    '''Function to train a model given the collected batch data set.'''

    if source_name is not None:
        source_network = m.MazePolicyNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy', map_location=torch.device(c.DEVICE)))
        source_network.eval()

        
        transfer_learning = True

    else:
        source_network = m.MazePolicyNetwork().to(c.DEVICE)
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
            replay_memory = torch.load(c.DATA_DIR + game_name + '_exploration_data.pt', map_location=torch.device(c.DEVICE))
            acc_reward_array = torch.load(c.DATA_DIR + game_name + '_rewards.pt', map_location=torch.device(c.DEVICE))
            loss_array = torch.load(c.DATA_DIR + game_name + '_loss.pt', map_location=torch.device(c.DEVICE))
            total_loss_array = torch.load(c.DATA_DIR + game_name + '_reg_loss.pt', map_location=torch.device(c.DEVICE))
            network_model = m.MazeFCNetwork().to(c.DEVICE)
            network_model.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_model', map_location=torch.device(c.DEVICE)))
            network_model.eval()
            target_net = m.MazeFCNetwork().to(c.DEVICE)
            target_net.load_state_dict(torch.load(c.MODEL_DIR + game_name + '_target_model', map_location=torch.device(c.DEVICE)))
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
            env = gym.make('gym_examples/GridWorld-v0', game=game, render_mode=render_mode)
            action_space = env.action_space
            obs = env.reset()
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

                    eps_decline_counter += 1

                if done:
                    print(f"Episode finished after {i+1} timesteps.")
                    break

            print(f'epsilon = {epsilon}')
            print(f'Accumulated a total reward of {accumulated_reward}.')
            acc_reward_array.append(accumulated_reward)
            time_steps_required_array.append(no_time_steps)

        env.close()

        for key in network_model.state_dict():
            source_network.state_dict()[key] = network_model.state_dict()[key]

        e.plot_nn_weights(network_model, game_name, c.PLOT_DIR)
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


    e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
                                          len(np.asarray(acc_reward_array))*no_segments,
                                          len(np.asarray(acc_reward_array)),
                                          c.NO_SEGMENTS,
                                          plot_dir=c.PLOT_DIR)
    e.plot_time_steps_in_maze(time_steps_required_array,
                              len(time_steps_required_array)*no_segments,
                              len(time_steps_required_array),
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


def train_a2c_network(game_name='onlycoincollecting', source_name=None, n_envs=1, transfer='lasso'):
    '''Train policy based on the advantage a2c framework.'''

    if source_name is not None:
        source_network = m.MazePolicyNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy', map_location=torch.device(c.DEVICE)))
        source_network.eval()
        network_policy = m.MazePolicyNetwork().to(c.DEVICE)
        # network_policy.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy', map_location=torch.device(c.DEVICE)))
        # network_policy.eval()
        network_value_function = m.MazeValueNetwork().to(c.DEVICE)
        # network_policy.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy', map_location=torch.device(c.DEVICE)))
        # network_policy.eval()
        
        transfer_learning = True

    else:
        source_network = m.MazePolicyNetwork().to(c.DEVICE)
        source_network.init_weights_to_zero()
        network_value_function = m.MazeValueNetwork().to(c.DEVICE)
        network_policy = m.MazePolicyNetwork().to(c.DEVICE)
        transfer_learning = False

    for param in source_network.parameters():
            param.requires_grad = False

    acc_reward_array = []
    loss_array = []
    total_loss_array = []
    game = GAME_TASKS[game_name]
    env_fns = [make_env(i, game) for i in range(n_envs)]
    vec_env = SubprocVecEnv(env_fns)
    action_space = vec_env.action_space
    obs = vec_env.reset()
    # env.close()
    state = create_fc_state_vector_mult_proc(obs, n_envs)
    accumulated_reward = np.zeros(n_envs)
    replay_memory = deque([], maxlen=c.CAPACITY)
    batch = deque([], maxlen=c.CAPACITY)

    for i in range(c.TIME_STEPS):

        vec_env.render()
        # time.sleep(0.1)
        # action = env.action_space.sample()
        term_bool = np.ones(n_envs)
        action = select_action(network_policy(state),
                                0,
                                False,
                                replay_memory,
                                action_space,
                                n_envs=n_envs)
        next_obs, reward, done, infos = vec_env.step(action)
        # print(reward)
        next_state = create_fc_state_vector_mult_proc(next_obs, n_envs)

        for k in range(n_envs):
            if done[k]:
                term_bool[k] = 0
            batch.append(o.Transition(state[k],
                                      action[k],
                                      next_state[k],
                                      reward[k],
                                      term_bool[k]))

        replay_memory.append(o.Transition(state,
                                          action,
                                          next_state,
                                          reward,
                                          term_bool))
        accumulated_reward += np.asarray(reward)
        state = next_state

        if ((i + 1) % c.BATCH_SIZE)==0:
            replay_memory_batch = o.Transition(*zip(*replay_memory))
            reward_batch = torch.from_numpy(np.asarray(replay_memory_batch.reward)).to(c.DEVICE)
            term_batch = torch.from_numpy(np.asarray(replay_memory_batch.terminated)).to(c.DEVICE)
            q_values = o.calculate_q_value_batch(term_batch,
                                                 network_value_function,
                                                 reward_batch,
                                                 next_state,
                                                 c.BATCH_SIZE,
                                                 n_envs,
                                                 gamma=c.GAMMA)

            loss_value, total_loss = o.a2c_optimization(network_policy,
                                                        network_value_function,
                                                        source_network,
                                                        batch,
                                                        q_values,
                                                        learning_rate=c.LEARNING_RATE,
                                                        lamb_lasso=c.LAMB_LASSO/(c.BATCH_SIZE*n_envs),
                                                        lamb_ridge=c.LAMB_RIDGE,
                                                        transfer_learning=transfer_learning,
                                                        transfer=transfer)
            replay_memory = deque([], maxlen=c.CAPACITY)
            batch = deque([], maxlen=c.CAPACITY)
            loss_array.append(loss_value)
            total_loss_array.append(total_loss)
            acc_reward_array.append(accumulated_reward)
            print(accumulated_reward)
            accumulated_reward = np.zeros(n_envs)
            total_loss = 0
            loss_value = 0

        if ((i + 1) % c.BATCH_SIZE*100) == 0:
            print('Save Plotdata and Model')
            np.save(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{c.LAMB_LASSO}_transfer_{transfer}', np.asarray(acc_reward_array).flatten())
            np.save(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_loss_a2c_residual_lamb_{c.LAMB_LASSO}_transfer_{transfer}', loss_array)
            torch.save(network_value_function.state_dict(), c.MODEL_DIR + game_name + '_value')
            torch.save(network_policy.state_dict(), c.MODEL_DIR + game_name + '_policy')

    vec_env.close()


def create_fc_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.zeros(5, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, observation['agent'][0], observation['agent'][1]] = 1
    # state_vector[1, observation['target'][0], observation['target'][1]] = 1

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


def create_fc_state_vector_mult_proc(observation, n_envs):
    '''Convert Observation output from environmnet into a state variable for regular NN
       with parallelization onto cpu cores.'''

    # print(observation['coins'])
    state_vector = torch.zeros(5, c.SIZE, c.SIZE, n_envs).to(c.DEVICE)
    state_vector[0] =  torch.from_numpy(observation['agent']).to(c.DEVICE).T
    state_vector[1] =  torch.from_numpy(observation['target']).to(c.DEVICE).T
    state_vector[2] =  torch.from_numpy(observation['coins']).to(c.DEVICE).T
    state_vector[3] =  torch.from_numpy(observation['traps']).to(c.DEVICE).T
    state_vector[4] =  torch.from_numpy(observation['walls']).to(c.DEVICE).T
    # state_vector[1, observation['target'][0], observation['target'][1]] = 1

    # for i in range(n_envs):
    #     state_vector[0, :, :, i] = torch.from_numpy(observation['agent'])[i]
    #     state_vector[1, :, :, i] = torch.from_numpy(observation['target'])[i]
    #     state_vector[2, :, :, i] = torch.from_numpy(observation['coins'])[i]
    #     state_vector[3, :, :, i] = torch.from_numpy(observation['traps'])[i]
    #     state_vector[4, :, :, i] = torch.from_numpy(observation['walls'])[i]

    state_vector = torch.reshape(state_vector, (5 * c.SIZE * c.SIZE, n_envs))

    return state_vector.T


def create_conv_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for convolutional NN.'''

    state_vector = torch.zeros(1, 3, c.SIZE, c.SIZE).to(c.DEVICE)
    state_vector[0, 0, observation['agent'][0], observation['agent'][1]] = 1
    state_vector[0, 1, observation['target'][0], observation['target'][1]] = 1
    state_vector[0, 2, observation['coins'][0], observation['coins'][1]] = 1
    state_vector[0, 3, observation['traps'][0], observation['traps'][1]] = 1
    state_vector[0, 4, observation['walls'][0], observation['walls'][1]] = 1

    return state_vector


def select_action(network_output, epsilon, mc_explore, state_history, action_space, stochastic=True, n_envs=1):
    '''Calculate, which action to take, with best estimated chance.'''

    thresholds = np.random.sample(size=n_envs)
    actions = np.zeros(n_envs, dtype=int)

    for i in range(n_envs):
        if thresholds[i] < epsilon:

            # if mc_explore:
            #     action = monte_carlo_exploration(state_history, action_space) ADAPT FOR MULTI-ENVIRONMENT

            actions[i] = np.random.randint(c.GRID_OUTPUT)

        else:
            if torch.max(network_output).isnan():
                print('Nan Value detected')

        # action = int(torch.argmax(network_output))
            if stochastic:
                prob_action = torch.nn.functional.softmax(torch.flatten(network_output[i]), dim=0)
                actions[i] = np.random.choice(np.arange(action_space.n),
                                        p=prob_action.cpu().detach().numpy())

            else:
                # action = np.random.choice(np.arange(4), p=prob_action.cpu().detach().numpy())
                actions[i] = np.argmax(torch.flatten(network_output).cpu().detach().numpy())

    return actions


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
