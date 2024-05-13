import config as c
import models as m
import training as t


def main():
    '''Main process of the biased RL setting.'''

    if c.FCMODEL:
        q_network = m.MazeFCNetwork().to(c.DEVICE)
        target_network = m.MazeFCNetwork().to(c.DEVICE)
    else:
        q_network = m.MazeFCNetwork().to(c.DEVICE)
        target_network = m.MazeFCNetwork().to(c.DEVICE)
    t.train_network(q_network, target_network, game='gym_examples/GridWorld-v0')

if __name__ == '__main__':
    main()
