import os
import numpy as np
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
from maze_project import config_maze as c
import evaluation as e


def plot_baselines():
    '''Main Script to plot saved reward  and loss data.'''

    game_name = 'coincollecting'
    source_name = 'pathfinding'
    transfer = 'group_lasso'

    coincollecting_reward_arrays = [# np.load(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{1}.npy'),
                                    np.load(c.DATA_DIR+game_name+f'_source_{None}_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_None.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{1}_transfer_group_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{0.1}_transfer_group_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_{source_name}_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_group_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_group_lasso.npy'),
                                    np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{0.1}_transfer_lasso.npy'),
                                    np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{1}_transfer_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{0.1}_transfer_group_lasso.npy'),
                                    # np.load(c.DATA_DIR+game_name+f'_source_onlycoincollecting_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_group_lasso.npy')
                                    ]
    
    onlycoincollecting_reward_arrays = [np.load(f'{c.DATA_DIR}onlycoincollecting_source_None_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_None.npy')]
    onlytraps_reward_arrays = [np.load(f'{c.DATA_DIR}onlytraps_source_None_accumulated_rmr_a2c_residual_lamb_{1}_transfer_None.npy')]

    pathfinding_reward_arrays = [np.load(f'{c.DATA_DIR}pathfinding_source_None_accumulated_rmr_a2c_residual_lamb_{1}.npy'),
                                 np.load(f'{c.DATA_DIR}pathfinding_source_None_accumulated_rmr_a2c_residual_lamb_{0.1}.npy'),
                                #  np.load(f'{c.DATA_DIR}pathfinding_source_None_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_None.npy'),
                                 np.load(f'{c.DATA_DIR}pathfinding_source_None_accumulated_rmr_a2c_residual_lamb_{0.0}.npy')]
    
    trapavoiding_reward_arrays = [np.load(f'{c.DATA_DIR}trapavoiding_source_None_accumulated_rmr_a2c_residual_lamb_{0.01}_transfer_None.npy'),
                                  np.load(f'{c.DATA_DIR}trapavoiding_source_pathfinding_accumulated_rmr_a2c_residual_lamb_{1}_transfer_lasso.npy'),
                                  np.load(f'{c.DATA_DIR}trapavoiding_source_onlytraps_accumulated_rmr_a2c_residual_lamb_{1}_transfer_lasso.npy')]

    maze_reward_arrays = [np.load(f'{c.DATA_DIR}mazenavigating_source_None_accumulated_rmr_a2c_residual_lamb_{1}_transfer_None.npy')]

    coincollecting_legends_array = [# f'Lasso Transfer path finding with \lambda=1',
                                    'No Transfer',
                                    # f'Group Lasso Transfer path finding with \lambda=1.0',
                                    # f'Group Lasso Transfer path finding with \lambda=0.1',
                                    # f'Group Lasso Transfer path finding with \lambda=0.01',
                                    # f'Lasso Transfer only coins with \lambda=0.01',
                                    # f'Group Lasso Transfer only coins with \lambda=0.01',
                                    f'Lasso Transfer only coins with \lambda=0.1',
                                    f'Lasso Transfer only coins with \lambda=1.0',
                                    # f'Group Lasso Transfer only coins with \lambda=0.1',
                                    # f'Group Lasso Transfer only coins with \lambda=0.01'
                                    ]
    
    maze_legends_array = ['No Transfer',
                                    ]

    onlycoincollecting_legends_array = ['No Transfer only coins']
    onlytraps_legends_array = ['No Transfer only traps']

    pathfinding_legends_array = ['No Transfer with \lambda=1.0',
                                 'No Transfer with \lambda=0.1',
                                #  'No Transfer with \lambda=0.01',
                                 'No Transfer with \lambda=0.0']
    
    trapavoiding_legends_array = ['No Transfer',
                                  'Transfer from Pathfinding with \lambda=1.0',
                                  'Transfer from only Traps with \lambda=1.0']

    e.plot_multiple_moving_average_rewards(coincollecting_reward_arrays,
                                           coincollecting_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix=game_name+'_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=320,
                                           x_lim=12000)
    
    e.plot_multiple_moving_average_rewards(onlycoincollecting_reward_arrays,
                                           onlycoincollecting_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='onlycoincollecting_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=320)
    
    e.plot_multiple_moving_average_rewards(onlytraps_reward_arrays,
                                           onlytraps_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='onlytraps_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=320)
    
    e.plot_multiple_moving_average_rewards(pathfinding_reward_arrays,
                                           pathfinding_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='pathfinding_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR)
    
    e.plot_multiple_moving_average_rewards(trapavoiding_reward_arrays,
                                           trapavoiding_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='trapavoiding_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=320,
                                           x_lim=12000)
    
    e.plot_multiple_moving_average_rewards(maze_reward_arrays,
                                           maze_legends_array,
                                           x_label='Episodes',
                                           y_label='Accumulated mean reward per episode',
                                           plot_suffix='maze_A2C_accumulated_running_mean_reward_maze_tasks',
                                           plot_dir=c.PLOT_DIR,
                                           mean_length=320,
                                           x_lim=12000)

    # e.plot_loss_function(loss_array,
    #                      len(loss_array),
    #                      plot_suffix=game_name+'_loss_evolution_a2c',
    #                      plot_dir=c.PLOT_DIR)
