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
        tag_labels=train_dataset.tag_labels,
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

    learner = JointCoBERTaLearner(
        model_type='phobert',
        model_name_or_path='vinai/phobert-base', 
        intent_loss_coef=0.4, 
        tag_loss_coef=0.6,
        use_intent_context_concat=False,
        use_intent_context_attention=False,
        attention_embedding_dim=200,
        max_seq_len=50,
        intent_embedding_type='hard',
        use_attention_mask=False,
        # device='cpu'
    )
    learner.train(
        train_dataset,
        test_dataset,
        train_batch_size=128,
        eval_batch_size=256,
        learning_rate=3e-5,
        n_epochs=1,
        view_model=True,
        monitor_test=True,
        save_best_model=True,
        model_dir='./models',
        model_name='run-phobert-ks',
        gpu_id=0
    )

test_training()