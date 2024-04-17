import config as c
import models as m
import training as t


def main():
    '''Main process of the biased RL setting.'''

    if c.FCMODEL:
        q_network = m.FullConnectedNetwork().to(c.DEVICE)
        target_network = m.FullConnectedNetwork().to(c.DEVICE)
    else:
        q_network = m.ConvNetwork().to(c.DEVICE)
        target_network = m.ConvNetwork().to(c.DEVICE)
    t.train_network(q_network, target_network)

if __name__ == '__main__':
    main()
