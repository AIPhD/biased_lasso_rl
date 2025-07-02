import config as c
import models as m
from cartpole_project import training_cartpole as tc
from maze_project import training_maze as tm
from racing_project import racing_training as tr
from atari_project import training_atari as ta



def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    # tc.train_a2c_network()
    # tm.train_network(conv_net=False)
    tm.train_a2c_network('mazenavigating', source_name=None)
    # tr.train_network()
    # ta.train_network()
    # ta.train_a2c_network('Alien')

if __name__ == '__main__':
    main()
