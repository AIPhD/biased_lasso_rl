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

def train_network(network_model,
                  target_net,
                  render_mode=c.RENDER):
    '''Function to train a model given the collected batch data set.'''

    # env = gym.make('Pong-v0')
    # wrapped_env = RelativePosition(env)
    if c.LOAD_EXPLORATION:
        replay_memory = torch.load(c.DATA_DIR + 'exploration_data.pt')
    else:
        replay_memory = deque([], maxlen=c.CAPACITY)

    target_update_counter = 0
    exploration_counter = 0
    eps_decline_counter = 0
    acc_reward_array = []

    env = gym.make('CartPole-v1', render_mode=render_mode)
    action_space = env.action_space
    for episode in range(c.EPISODES):
        obs, _ = env.reset()
        # env.close()
        state = create_fc_state_vector(obs)
        print(f"{episode} episodes done.")
        accumulated_reward = 0

        for i in range(c.TIME_STEPS):
            env.render()
            # time.sleep(0.1)
            # action = env.action_space.sample()
            epsilon = max(1 - 0.9 * eps_decline_counter/c.EPS_DECLINE, 0.05)
            exploration_counter += 1
            action = select_action(network_model(state),
                                   epsilon)
            next_obs, reward, done, _, _ = env.step(action)

            if done:
                # next_state=None
                reward = -10

            else:
                reward = i

            accumulated_reward = i

            next_state = create_fc_state_vector(next_obs)
            replay_memory.append(o.Transition(state, action, next_state, reward))
            state = next_state
            n_segments = int(len(replay_memory)/c.BATCH_SIZE)

            if len(replay_memory) >= c.BATCH_SIZE:
                o.optimization_step(network_model,
                                    target_net,
                                    replay_memory,
                                    n_segments,
                                    c.GAMMA,
                                    c.LEARNING_RATE,
                                    c.LAMB,
                                    c.BATCH_SIZE)
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

    state_vector = torch.Tensor(observation).to(c.DEVICE)
    return state_vector


def select_action(network_output, epsilon, stochastic_selection=True):
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
            action = int(torch.argma(network_output))

    return action

# %%
