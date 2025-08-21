# import time
#%%
from collections import deque
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import SubprocVecEnv
from cartpole_project import config_cartpole as c
import optimization as o
import models as m
import evaluation as e


def make_env(seed):
    def _init():
        env = gym.make('CartPole-v1',
                       render_mode=c.RENDER)
        return env
    return _init

def train_dqn_network(render_mode=c.RENDER):
    '''Function to train a model given the collected batch data set.'''

    network_model = m.FullConnectedNetwork().to(c.DEVICE)
    target_net = m.FullConnectedNetwork().to(c.DEVICE)

    for param in target_net.parameters():
        param.requires_grad = False

    replay_memory = deque([], maxlen=c.CAPACITY)
    source_network = m.FullConnectedNetwork().to(c.DEVICE)
    source_network.init_weights_to_zero()
    transfer_learning = False
    target_update_counter = 0
    acc_reward_array = []
    env = gym.make('CartPole-v1', render_mode=render_mode)
    exploration_counter = 0
    loss_array = []
    total_loss_array = []
    epsilon = 1

    for episode in range(c.EPISODES):
        obs, _ = env.reset()
        state = create_fc_state_vector(obs)
        print(f"{episode} episodes done.")
        accumulated_reward = 0

        for i in range(c.TIME_STEPS):
            env.render()
            # time.sleep(0.1)
            # action = env.action_space.sample()

            if exploration_counter >= c.EXPLORATION:

                    epsilon = max(c.EPS_MIN, epsilon * c.EPS_DECLINE_FACTOR)

            term_bool = 1
            action = select_action(network_model(state), epsilon)
            next_obs, reward, done, _, _ = env.step(action)

            if done:
                term_bool = 0
                reward = 1

            accumulated_reward = i
            next_state = create_fc_state_vector(next_obs)
            replay_memory.append(o.Transition(state, action, next_state, 1*reward, term_bool))
            state = next_state

            if len(replay_memory) >= c.BATCH_SIZE:
                loss_value, total_loss = o.optimization_step(network_model,
                                                             target_net,
                                                             replay_memory,
                                                             c.GAMMA,
                                                             c.LEARNING_RATE,
                                                             c.MOMENTUM,
                                                             c.LAMB_LASSO,
                                                             c.LAMB_RIDGE,
                                                             c.BATCH_SIZE,
                                                             source_network=source_network,
                                                             transfer_learning=transfer_learning)
                target_update_counter += 1
                loss_array.append(loss_value)
                total_loss_array.append(total_loss)

                if target_update_counter == c.UPDATE_TARGET:

                    for key in target_net.state_dict():
                        target_net.state_dict()[key] = c.TAU*network_model.state_dict()[key] + (1 - c.TAU)*target_net.state_dict()[key]

                    target_update_counter = 0

            exploration_counter += 1

            if done:
                print(f"Episode finished after {i+1} timesteps.")
                break

        print(f'epsilon = {epsilon}')
        acc_reward_array.append(accumulated_reward)
        env.close()

    e.plot_cumulative_rewards_per_segment(np.asarray(acc_reward_array),
                                          c.EPISODES * 1,
                                          c.EPISODES,
                                          1,
                                          plot_dir=c.PLOT_DIR)
    e.plot_loss_function(loss_array,
                         len(loss_array),
                         plot_dir=c.PLOT_DIR)
    e.plot_loss_function(total_loss_array,
                         len(total_loss_array),
                         y_label='Regularized Loss',
                         plot_suffix='total_loss',
                         plot_dir=c.PLOT_DIR)
    e.plot_nn_weights(network_model, 0, c.PLOT_DIR)

    return network_model


def train_a2c_network(game='CartPole-v1', source_name=None, n_envs=1):
    '''Train policy based on the advantage a2c framework.'''

    if source_name is not None:
        source_network = m.CartpolePolicyNetwork().to(c.DEVICE)
        source_network.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy'))
        source_network.eval()
        network_policy = m.CartpolePolicyNetwork().to(c.DEVICE)
        network_policy.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy'))
        network_policy.eval()
        network_value_function = m.CartpoleValueNetwork().to(c.DEVICE)
        # network_policy.load_state_dict(torch.load(c.MODEL_DIR + source_name + '_policy'))
        # network_policy.eval()
        
        transfer_learning = True

    else:
        source_network = m.CartpolePolicyNetwork().to(c.DEVICE)
        source_network.init_weights_to_zero()
        network_value_function = m.CartpoleValueNetwork().to(c.DEVICE)
        network_policy = m.CartpolePolicyNetwork().to(c.DEVICE)
        transfer_learning = False

    acc_reward_array = []
    loss_array = []
    total_loss_array = []

    for episode in range(c.EPISODES):

        # env = gym.make('gym_examples/GridWorld-v0', game=game, render_mode=c.RENDER)
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
            next_obs, reward, done, _ = vec_env.step(action)
            print(reward)
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

            if i==c.TIME_STEPS-1:
                replay_memory_batch = o.Transition(*zip(*replay_memory))
                reward_batch = torch.from_numpy(np.asarray(replay_memory_batch.reward)).to(c.DEVICE)
                term_batch = torch.from_numpy(np.asarray(replay_memory_batch.terminated)).to(c.DEVICE)
                q_values = o.calculate_q_value_batch(term_batch,
                                                     network_value_function,
                                                     reward_batch,
                                                     next_state,
                                                     c.TIME_STEPS,
                                                     n_envs,
                                                     gamma=c.GAMMA)

        loss_value, total_loss = o.a2c_optimization(network_policy,
                                                    network_value_function,
                                                    source_network,
                                                    batch,
                                                    q_values,
                                                    learning_rate=c.LEARNING_RATE,
                                                    lamb_lasso=c.LAMB_LASSO/np.sqrt(len(replay_memory)),
                                                    lamb_ridge=c.LAMB_RIDGE/np.sqrt(len(replay_memory)),
                                                    transfer_learning=transfer_learning)

        loss_array.append(loss_value)
        total_loss_array.append(total_loss)
        print(f"{episode + 1} episode(s) done.")
        print(f'Accumulated a total reward of {accumulated_reward}.')
        acc_reward_array.append(accumulated_reward)

        if (episode + 1) % c.SAVE_PERIOD == 0:

            print('Save Plots and Model')
            e.plot_moving_average_reward(np.asarray(acc_reward_array).flatten(),
                                        len(np.asarray(acc_reward_array).flatten()),
                                        1,
                                        plot_suffix=f'CartPole_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{c.LAMB_LASSO}',
                                        plot_dir=c.PLOT_DIR,
                                        transfer_learning=transfer_learning)
            e.plot_loss_function(loss_array,
                                 len(loss_array),
                                 plot_suffix='CartPole_loss_evolution_a2c',
                                 plot_dir=c.PLOT_DIR)
            torch.save(network_value_function.state_dict(), c.MODEL_DIR + 'CartPole_value')
            torch.save(network_policy.state_dict(), c.MODEL_DIR + 'CartPole_policy')
            # e.plot_loss_function(total_loss_array,
            #                      len(total_loss_array),
            #                      y_label='Regularized Loss',
            #                      plot_suffix=game_name+'_total_loss_evolution_a3c',
            #                      plot_dir=c.PLOT_DIR)

    vec_env.close()


def create_fc_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.Tensor(observation).to(c.DEVICE)
    return normalize_state(state_vector)


def create_fc_state_vector_mult_proc(observation, n_env):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.Tensor(observation).to(c.DEVICE)                   # CHECK DIMENSIONS
    return normalize_state(state_vector)


def select_action(network_output, numb_actions, epsilon, stochastic_selection=True):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:
        if torch.max(network_output).isnan():
            print('Nan Value detected')

        if stochastic_selection:
            prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
            action = np.random.choice(np.arange(numb_actions),
                                      p=prob_action.cpu().detach().numpy())

        else:
            action = int(torch.argmax(network_output))
            # print(action)

    else:
        action = np.random.randint(numb_actions)

    return action

def normalize_state(state):
    '''Normalize state variable before the optimization step.'''

    # state[0] /= 4.8
    # state[1] /= 2.5
    # state[2] /= 0.418
    # state[3] /= 0.3

    return state
    