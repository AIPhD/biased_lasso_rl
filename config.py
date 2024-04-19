from pathlib import Path
import torch


DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

SIZE = 9
INPUT = 4
OUTPUT = 2
EPOCHS = 3000
TAU = 0.05
EPISODES = 250
HIDDEN_NODE_COUNT = 16
TRAIN_EPOCH = 10
GAMMA = 0.99
LAMB = 0.0
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CAPACITY = 10000
BATCH_SIZE = 16
UPDATE_TARGET = 1
RENDER = 'rgb_array'
MINI_BATCH_TRAINING = False
FCMODEL = True
SAVE_PLOT = False
MC_EXPLORE_CONST = 1
EXPLORATION = 5000
EPS_DECLINE = 10000
LOAD_EXPLORATION = False
SAVE_EXPLORATION = True
DATA_DIR = 'collected_data/'
PLOT_DIR = '/home/steven/biased_lasso_rl/plots/'
Path("collected_data").mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 