from pathlib import Path
import torch
import config as c

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

CART_INPUT = 4
CART_OUTPUT = 2
EPISODES = 500
TAU = 0.01
TIME_STEPS = 600
HIDDEN_NODE_COUNT = 128
TRAIN_EPOCH = 10
GAMMA = 0.99
LAMB = 1.0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CAPACITY = 10000
BATCH_SIZE = 200
UPDATE_TARGET = 1
RENDER = 'rgb_array'
MINI_BATCH_TRAINING = False
SAVE_PLOT = False
EXPLORATION = 500
EPS_DECLINE = 100
NO_SEGMENTS = 1
LOAD_EXPLORATION = False
SAVE_EXPLORATION = True
DATA_DIR = 'cartpole_project/collected_data/'
PLOT_DIR = 'cartpole_project/plots/'
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)
Path(PLOT_DIR).mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 