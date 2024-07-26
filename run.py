import config as c
import models as m
from cartpole_project import training_cartpole as tc
from maze_project import training_maze as tm


def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    # cartpole_q_network = m.FullConnectedNetwork().to(c.DEVICE)
    # cartpole_target_network = m.FullConnectedNetwork().to(c.DEVICE)
    # tc.train_network(cartpole_q_network, cartpole_target_network)
    # else:
    q_network = m.MazeFCNetwork().to(c.DEVICE)
    target_network = m.MazeFCNetwork().to(c.DEVICE)
    tm.train_network(q_network, target_network, game='gym_examples/GridWorld-v0')

if __name__ == '__main__':
    main()
