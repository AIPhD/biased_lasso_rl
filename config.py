import torch

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

INPUT = 50
OUTPUT = 4
EPOCHS = 200
EPISODES = 100
TRAIN_EPOCH = 10
GAMMA = 0.1
CAPACITY = 300
BATCH_SIZE = 100
UPDATE_TARGET = 50
RENDER = 'human' # 'rgb_array'
 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0 when looking for virtual display output in WSL2 or copy in bashrc file