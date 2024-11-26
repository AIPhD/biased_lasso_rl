from pathlib import Path
import config as c

DEVICE = c.DEVICE
EPISODES = 500
TAU = 1
TIME_STEPS = 10000
TRAIN_EPOCH = 10
GAMMA = 0.99
LAMB_LASSO = 0.0
LAMB_RIDGE = 0.0
LEARNING_RATE = 0.00025
MOMENTUM = 0.95
CAPACITY = 100000
BATCH_SIZE = 32
UPDATE_TARGET = 1000
UPDATE_Q = 4
FRAME_SKIP = 4
RENDER = 'rgb_array'
SAVE_PLOT = False
EXPLORATION = 50000
EPS_DECLINE = 200000
SAVE_SEGMENT = True
LOAD_SEGMENT = False
LOAD_NETWORK = False
DATA_DIR = 'atari_project/data/'
PLOT_DIR = 'atari_project/plots'
MODEL_DIR = 'atari_project/saved_models/'

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 