from pathlib import Path
import config as c

DEVICE = c.DEVICE
EPISODES = 10
TAU = 1
TIME_STEPS = 400
TRAIN_EPOCH = 10
GAMMA = 0.99
LAMB_LASSO = 1.0
LAMB_RIDGE = 1.0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CAPACITY = 20000
BATCH_SIZE = 256
UPDATE_TARGET = 4096
UPDATE_Q = 4
RENDER = 'rgb_array'
SAVE_PLOT = False
MC_EXPLORE_CONST = 1
EXPLORATION = 0
EPS_DECLINE = 4096
NO_SEGMENTS = 5
LOAD_EXPLORATION = False
SAVE_EXPLORATION = False
SAVE_NETWORK = False
LOAD_NETWORK = False
DATA_DIR = 'collected_data/'
PLOT_DIR = 'racing_project/plots'
MODEL_DIR = 'racing_project/saved_models/'

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 