
import random
import os

import torch
import numpy as np

def seed_everything(seed=1234):
    print(f'SET RANDOM SEED = {seed}')
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)