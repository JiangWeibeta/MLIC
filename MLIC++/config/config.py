import torch.nn as nn
from utils.utils import Config

def model_config():
    config = Config({
        "N": 192,
        "M": 320,
        "slice_num": 10,
        "context_window": 5,
        "act": nn.GELU,
    })

    return config
