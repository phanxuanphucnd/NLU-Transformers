# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from arizona.nlu.datasets import JointNLUDataset
from arizona.nlu.learners.joint import JointCoBERTaLearner


def test_evaluate():

    test_path = 'data/kcloset/test.csv'
    model_path = 'models/phobert-nlu'

    learner = JointCoBERTaLearner(model_name_or_path='phobert')
    learner.load_model(model_path)
    
    test_dataset = JointNLUDataset(
        mode='test',
        data_path=test_path,
        tokenizer='phobert',
        text_col='text',
        intent_col='intent',
        tag_col='tags',
        intent_labels=learner.intent_labels,
        tag_labels=learner.tag_labels,
        special_intents=["UNK"],
        special_tags=["PAD", "UNK"],
        max_seq_len=50,
        ignore_index=0,
        lowercase=True,
        rm_emoji=False,
        rm_url=False,
        rm_special_token=False,
        balance_data=False
    )

    out = learner.evaluate(test_dataset, batch_size=256, view_report=True)

test_evaluate()