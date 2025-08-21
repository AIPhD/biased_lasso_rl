from pathlib import Path
import torch
import config as c

DEVICE = c.DEVICE
SIZE = 7
GRID_INPUT = SIZE**2*5
GRID_OUTPUT = 4
EPISODES = 300
CONV_KERNEL = 2
POOL_KERNEL = 3
TAU = 1
TIME_STEPS = 200000
HIDDEN_NODE_COUNT = 10*GRID_INPUT
TRAIN_EPOCH = 10
GAMMA = 0.9
LAMB_LASSO = 1*10**(-2)
LAMB_RIDGE = 0*10**(-6)
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
CAPACITY = 500000
BATCH_SIZE = 20 #4096
UPDATE_Q = 1
UPDATE_TARGET = 1024
RENDER = 'rgb_array'
MINI_BATCH_TRAINING = False
CONVMODEL = False
SAVE_PLOT = False
MC_EXPLORE_CONST = 1
EXPLORATION = 10000
EPS_DECLINE = 4096
NO_SEGMENTS = 1
LOAD_NETWORK = False
LOAD_SEGMENT = False
SAVE_SEGMENT = True
SAVE_PERIOD = 50
DATA_DIR = 'maze_project/data/'
PLOT_DIR = 'maze_project/plots'
MODEL_DIR = 'maze_project/saved_models/'
Path("collected_data").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 