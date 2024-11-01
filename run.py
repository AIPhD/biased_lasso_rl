import config as c
import models as m
from cartpole_project import training_cartpole as tc
from maze_project import training_maze as tm
from racing_project import racing_training as tr
from atari_project import training_atari as ta



def main():
    '''Main process of the biased RL setting for arbitrary environments.'''

    # cartpole_q_network = m.FullConnectedNetwork().to(c.DEVICE)
    # cartpole_target_network = m.FullConnectedNetwork().to(c.DEVICE)

    # carracing_network = m.RacingNetwork().to(c.DEVICE)
    # carracing_target_network = m.RacingNetwork().to(c.DEVICE)

    # for param in cartpole_target_network.parameters():
    #     param.requires_grad = False

    # for param in carracing_target_network.parameters():
    #     param.requires_grad = False

    # tc.train_network(cartpole_q_network, cartpole_target_network)
    tm.train_network(conv_net=False)
    # tr.train_network(carracing_network, carracing_target_network)
    # ta.train_network()

if __name__ == '__main__':
    main()
