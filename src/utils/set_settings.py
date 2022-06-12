import torch
import random
import numpy as np


def set_seed(n_seed):
    np.random.seed(n_seed)
    random.seed(n_seed)
    torch.manual_seed(n_seed)
    torch.cuda.manual_seed(n_seed)
    torch.backends.cudnn.deterministic = True