import os
import config as c
import models as m
import evaluation as e
from cartpole_project import training_cartpole as tc
from maze_project import training_maze_old as tm
from maze_project import rl_maze as rlm
from atari_project import rl_atari as rla
from atari_project import training_atari_old as ta



def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    n_envs = os.cpu_count()

    # dqn_sim_maze = rlm.MazeDQNLearning('mazenavigating')
    # dqn_sim_maze.train_agent()

    # ddqn_sim_maze = rlm.MazeDDQNLearning('mazenavigating')
    # ddqn_sim_maze.train_agent()

    a2c_sim_maze = rlm.MazeA2CLearning('pathfinding', n_envs=8, batch_size=36)
    a2c_sim_maze.train_agent()

    # a2c_sim_atari = rla.AtariA2CLearning('Centipede', n_envs=n_envs)
    # a2c_sim_atari.train_agent()



if __name__ == '__main__':
    main()
