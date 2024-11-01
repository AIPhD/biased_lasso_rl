from pathlib import Path
import torch
import config as c

DEVICE = c.DEVICE
SIZE = 5
GRID_INPUT = SIZE**2*5
GRID_OUTPUT = 4
EPISODES = 10
CONV_KERNEL = 2
POOL_KERNEL = 3
TAU = 1
TIME_STEPS = 500
HIDDEN_NODE_COUNT = 3*GRID_INPUT
TRAIN_EPOCH = 10
GAMMA = 0.999
LAMB_LASSO = 5*10**(-7)
LAMB_RIDGE = 5*10**(-8)
LEARNING_RATE = 0.00005
MOMENTUM = 0.9
CAPACITY = 50000
BATCH_SIZE = 6
UPDATE_Q = 1
UPDATE_TARGET = 512
RENDER = 'human'
MINI_BATCH_TRAINING = False
CONVMODEL = False
SAVE_PLOT = False
MC_EXPLORE_CONST = 1
EXPLORATION = 0
EPS_DECLINE = 4096
NO_SEGMENTS = 1
LOAD_EXPLORATION = False
SAVE_EXPLORATION = False
SAVE_NETWORK = False
LOAD_NETWORK = False
LOAD_SEGMENT = False
SAVE_SEGMENT = True
DATA_DIR = 'maze_project/data/'
PLOT_DIR = 'maze_project/plots'
MODEL_DIR = 'maze_project/saved_models/'
Path("collected_data").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 