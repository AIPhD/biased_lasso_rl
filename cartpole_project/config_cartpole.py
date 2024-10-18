from pathlib import Path
import torch
import config as c

# DEVICE = (
#     "cuda"
#     if torch.cuda.is_available()
#     else "mps"
#     if torch.backends.mps.is_available()
#     else "cpu"
# )
DEVICE = c.DEVICE
CART_INPUT = 4
CART_OUTPUT = 2
EPISODES = 500
TAU = 1
TIME_STEPS = 600
HIDDEN_NODE_COUNT = 64
TRAIN_EPOCH = 10
GAMMA = 0.999
LAMB_LASSO = 0.0000
LAMB_RIDGE = 0.000
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
CAPACITY = 100000
BATCH_SIZE = 128
UPDATE_TARGET = 10000
RENDER = 'rgb_array'
MINI_BATCH_TRAINING = False
SAVE_PLOT = False
EXPLORATION = 5000
EPS_DECLINE_FACTOR = 0.999
NO_SEGMENTS = 1
LOAD_EXPLORATION = False
SAVE_EXPLORATION = False
DATA_DIR = 'cartpole_project/collected_data/'
PLOT_DIR = 'cartpole_project/plots/'

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 