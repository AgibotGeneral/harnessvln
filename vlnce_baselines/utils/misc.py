import os
import random

import numpy as np
import torch


def seed_everything(seed: int):
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(device_number):
    device = torch.device("cpu")
    if device_number >= 0 and torch.cuda.is_available():
        device = torch.device("cuda:{0}".format(device_number))
    return device
