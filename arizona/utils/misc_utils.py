# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import random
import numpy as np

from typing import Any
from transformers import AutoTokenizer, RobertaConfig

from arizona.nlu.models import JointCoBERTa

CONFIGS_REGISTRY = {
    'coberta': RobertaConfig, 
    'phobert': RobertaConfig, 
}

MODELS_REGISTRY = {
    'coberta': JointCoBERTa,
    'phobert': JointCoBERTa, 
}

TOKENIZERS_REGISTRY = {
    'coberta': AutoTokenizer,
    'phobert': AutoTokenizer, 
}

MODEL_PATH_MAP = {
    'coberta': '',
    'phobert': 'vinai/phobert-base',
}

def str2bool(value):
    return str(value).lower() in ('yes', 'true', 't', '1')

def ifnone(a: Any, b: Any) -> Any:
    """a if a is not None, otherwise b."""
    return b if a is None else a

def get_from_registry(key, registry):
    if hasattr(key, 'lower'):
        key = key.lower()

    if key in registry:
        return registry[key]
    else:
        raise ValueError(f"Key `{key}` not supported, available options: {registry.keys()}")

def set_seed(SEED):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)