import os
import config as c
import models as m
import evaluation as e
from cartpole_project import training_cartpole as tc
from maze_project import training_maze as tm
from atari_project import training_atari as ta



def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    n_envs = os.cpu_count()
    # tc.train_a2c_network()
    # tm.train_dqn_network(conv_net=False)

    # tm.train_a2c_network('mazenavigating', source_name=None, n_envs=16, transfer=None)

    ta.train_a2c_network('Centipede', source_name=None, n_envs=16, transfer=None)
    # ta.train_a2c_baseline('SpaceInvaders')
    # ta.train_network()
    # ta.train_a2c_network('Alien')


if __name__ == '__main__':
    main()
