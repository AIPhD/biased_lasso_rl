import config as c
import models as m
from cartpole_project import training_cartpole as tc
from maze_project import training_maze as tm
from racing_project import racing_training as tr
from atari_project import training_atari_dqn as ta



def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    # tc.train_network()
    # tm.train_network(conv_net=False)
    # tr.train_network()
    # ta.train_network()
    ta.train_a3c_network('Centipede')

if __name__ == '__main__':
    main()
