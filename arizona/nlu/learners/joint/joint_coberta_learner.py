# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import torch

from transformers import PreTrainedTokenizerBase
from arizona.nlu.models.joint import JointCoBERTa
from arizona.nlu.datasets.joint_dataset import JointNLUDataset

class JointCoBERTaLearner():
    def __init__(
        self, 
        model: JointCoBERTa=None, 
        tokenizer: PreTrainedTokenizerBase=None, 
        device: str=None
    ):
        super(JointCoBERTaLearner, self).__init__()
        
        self.model = model
        self.tokenizer = tokenizer
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    @property
    def __name__(self):
        return 'JointCoBERTaNLU'

    def train(
        self, 
        train_dataset: JointNLUDataset=None,
        test_dataset: JointNLUDataset=None,
        train_batch_size: int=32,
        eval_batch_size: int=64,
        learning_rate: float=5e-5,
        n_epochs: int=10,
        weight_decay: float=0.0,
        gradient_acculation_steps: int=1,
        adam_epsilon: float=1e-8,
        max_grad_norm: float=1.0,
        max_steps: int=-1,
        warmup_steps: int=0,
        dropout: float=0.1,
        ignore_index: int=0,
        slot_loss_coef: float=1.0,
        **kwargs
    ):

        pass

    
    def evaluate(self):
        raise NotImplementedError

    def predict(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    