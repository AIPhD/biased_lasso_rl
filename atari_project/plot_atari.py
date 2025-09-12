import os
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from atari_project import config_atari as c
import evaluation as e


def plot_baselines():
    '''Main Script to plot saved reward  and loss data.'''

    centipede_reward_arrays = [np.load(f'{c.DATA_DIR}Centipede_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy')]
    centipede_legends_array = ['No Transfer']

    galaxian_reward_arrays = [np.load(f'{c.DATA_DIR}Galaxian_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy'),
                              np.load(f'{c.DATA_DIR}Galaxian_source_Centipede_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_group_lasso_kernel.npy')]
    galaxian_legends_array = ['No Transfer',
                              'Transfer from Centipede']

    phoenix_reward_arrays = [np.load(f'{c.DATA_DIR}Phoenix_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy')]
    phoenix_legends_array = ['No Transfer']

    pong_reward_arrays = [np.load(f'{c.DATA_DIR}Pong_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy')]
    pong_legends_array = ['No Transfer']

    alien_reward_arrays = [np.load(f'{c.DATA_DIR}Alien_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy')]
    alien_legends_array = ['No Transfer']

    invaders_reward_arrays = [np.load(f'{c.DATA_DIR}SpaceInvaders_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_None.npy'),
                              np.load(f'{c.DATA_DIR}SpaceInvaders_source_Centipede_accumulated_rmr_a2c_residual_lamb_{0.0}_transfer_group_lasso_kernel.npy')]
    invaders_legends_array = ['No Transfer',
                              'Transfer from Centipede']


    e.plot_multiple_moving_average_rewards(centipede_reward_arrays,
                                           centipede_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='centipede_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=25*c.BATCH_SIZE)

    e.plot_multiple_moving_average_rewards(pong_reward_arrays,
                                           pong_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='pong_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=25*c.BATCH_SIZE)

    e.plot_multiple_moving_average_rewards(alien_reward_arrays,
                                           alien_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='alien_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=25*c.BATCH_SIZE)
    
    e.plot_multiple_moving_average_rewards(galaxian_reward_arrays,
                                           galaxian_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='galaxian_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=16*c.BATCH_SIZE)
    e.plot_multiple_moving_average_rewards(phoenix_reward_arrays,
                                           phoenix_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='phoenix_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=16*c.BATCH_SIZE)
    e.plot_multiple_moving_average_rewards(invaders_reward_arrays,
                                           invaders_legends_array,
                                           x_label='episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='invaders_A2C_accumulated_running_mean_reward',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=16*c.BATCH_SIZE)
    