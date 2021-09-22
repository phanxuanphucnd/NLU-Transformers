# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from arizona.nlu.datasets import JointNLUDataset
from arizona.nlu.learners.joint import JointCoBERTaLearner


def test_training():
    train_dataset = JointNLUDataset(
        mode='train',
        data_path='data/cometv3/train.csv',
        tokenizer='phobert',
        text_col='text',
        intent_col='intent',
        tag_col='tags',
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

    test_dataset = JointNLUDataset(
        mode='test',
        data_path='data/cometv3/test.csv',
        tokenizer='phobert',
        text_col='text',
        intent_col='intent',
        tag_col='tags',
        intent_labels=train_dataset.intent_labels,
        tag_labels=train_dataset.intent_labels,
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

    learner = JointCoBERTaLearner(model_name_or_path='phobert')
    learner.train(
        train_dataset,
        test_dataset,
        train_batch_size=128,
        eval_batch_size=256,
        learning_rate=5e-5,
        n_epochs=100,
        logging_steps=200,
        save_steps=200,
        view_model=False,
        monitor_test=True,
        save_best_model=True,
        model_dir='./models',
        model_name='phobert.nlu'
    )

test_training()