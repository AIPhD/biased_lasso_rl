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
INPUT = 3*9*9
OUTPUT = 4
EPOCHS = 20000
EPISODES = 250
TRAIN_EPOCH = 10
GAMMA = 0.99
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CAPACITY = 100000
BATCH_SIZE = 32
UPDATE_TARGET = 512
RENDER = 'human' # 'rgb_array'
MINI_BATCH_TRAINING = False
FCMODEL = False
MC_EXPLORE_CONST = 1
EXPLORATION = 500
EPS_DECLINE = 10000
LOAD_EXPLORATION = False
SAVE_EXPLORATION = True
DATA_DIR = 'collected_data/'
Path("collected_data").mkdir(parents=True, exist_ok=True)

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0 when looking for virtual display output in WSL2 or copy in bashrc file