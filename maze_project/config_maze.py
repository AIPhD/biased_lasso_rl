from pathlib import Path
import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SIZE = 5
GRID_INPUT = SIZE**2*3
GRID_OUTPUT = 4
EPISODES = 200
CONV_KERNEL = 2
POOL_KERNEL = 3
TAU = 1
TIME_STEPS = 250
HIDDEN_NODE_COUNT = GRID_INPUT
TRAIN_EPOCH = 10
GAMMA = 0.99
LAMB = 1.0
LEARNING_RATE = 0.0001
MOMENTUM = 0.9
CAPACITY = 16384
BATCH_SIZE = 128
UPDATE_TARGET = 128
RENDER = 'rgb_array'
MINI_BATCH_TRAINING = False
CONVMODEL = False
SAVE_PLOT = False
MC_EXPLORE_CONST = 1
EXPLORATION = 4096
EPS_DECLINE = 4096 + 8192
NO_SEGMENTS = 1
LOAD_EXPLORATION = False
SAVE_EXPLORATION = False
SAVE_NETWORK = False
LOAD_NETWORK = False
DATA_DIR = 'collected_data/'
PLOT_DIR = 'maze_project/plots/'
MODEL_DIR = 'maze_project/saved_models/'
Path("collected_data").mkdir(parents=True, exist_ok=True)
Path("plots").mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 