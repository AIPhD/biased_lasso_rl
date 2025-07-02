from pathlib import Path
import torch


DEVICE = (
    "cpu"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

MINI_BATCH_TRAINING = False

 # use command DISPLAY=$(grep nameserver /etc/resolv.conf | awk '{print $2}'):0.0
 # when looking for virtual display output in WSL2 or copy in bashrc file
 