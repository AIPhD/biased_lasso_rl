import time
from gym_examples.envs.grid_world import GridWorldEnv
from gym_examples.wrappers import RelativePosition
import gym
import matplotlib.pyplot as plt
import numpy as np
import config as c
import models as m
import training as t


def main():
    '''Main function of the RL setting.'''
    q_network = m.ConvQNetwork().to(c.DEVICE)
    t.train_network(q_network)

if __name__ == '__main__':
    main()
