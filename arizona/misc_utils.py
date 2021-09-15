# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import random
import numpy as np

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)