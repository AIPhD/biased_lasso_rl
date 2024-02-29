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
EPOCHS = 30
GAMMA = 0.7
CAPACITY = 300
BATCH_SIZE = 50
RENDER = 'human' # 'rgb_array'
 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0 when looking for virtual display output