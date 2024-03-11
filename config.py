import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

INPUT = 128
OUTPUT = 4
EPOCHS = 20000
EPISODES = 250
TRAIN_EPOCH = 10
GAMMA = 0.99
LEARNING_RATE = 0.001
MOMENTUM = 0.9
CAPACITY = 100000
BATCH_SIZE = 128
UPDATE_TARGET = 512
RENDER = 'human' # 'rgb_array'
MINI_BATCH_TRAINING = False
FCMODEL = False
EXPLORATION = 5000
EPS_DECLINE = 100000

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0 when looking for virtual display output in WSL2 or copy in bashrc file