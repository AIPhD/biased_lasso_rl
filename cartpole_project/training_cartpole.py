# import time
#%%
from collections import deque
import torch
import gym
import numpy as np
from cartpole_project import config_cartpole as c
import optimization as o
import models as m
import evaluation as e

def train_network(render_mode=c.RENDER):
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


def create_fc_state_vector(observation):
    '''Convert Observation output from environmnet into a state variable for regular NN.'''

    state_vector = torch.Tensor(observation).to(c.DEVICE)
    return normalize_state(state_vector)


def select_action(network_output, epsilon, stochastic_selection=False):
    '''Calculate, which action to take, with best estimated chance.'''

    threshold = np.random.sample()

    if threshold < epsilon:
        action = np.random.randint(c.CART_OUTPUT)

    else:
        if torch.max(network_output).isnan():
            print('Nan Value detected')

        if stochastic_selection:
            prob_action = torch.nn.functional.softmax(torch.flatten(network_output), dim=0)
            action = np.random.choice(np.arange(2), p=prob_action.cpu().detach().numpy())

        else:
            action = int(torch.argmax(network_output))
            # print(action)

    return action

def normalize_state(state):
    '''Normalize state variable before the optimization step.'''

    # state[0] /= 4.8
    # state[1] /= 2.5
    # state[2] /= 0.418
    # state[3] /= 0.3

    return state
    