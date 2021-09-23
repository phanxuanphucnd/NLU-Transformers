# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import random
import numpy as np

from typing import Any
from transformers import RobertaConfig, BertConfig
from transformers import AutoTokenizer, BertTokenizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report
from seqeval.metrics import ( 
    precision_score as seqeval_precision, recall_score as seqeval_recall, 
    f1_score as seqeval_f1, classification_report as seqeval_report
)

from arizona.nlu.models import JointCoBERTa

CONFIGS_REGISTRY = {
    'coberta': RobertaConfig, 
    'phobert': RobertaConfig, 
    'vinai/phobert-base': RobertaConfig,
    'vinai/phobert-large': RobertaConfig,
    'bert': BertConfig, 
}

MODELS_REGISTRY = {
    'coberta': JointCoBERTa,
    'phobert': JointCoBERTa, 
    'vinai/phobert-base': JointCoBERTa,
    'vinai/phobert-large': JointCoBERTa,
    'bert': JointCoBERTa,
}

TOKENIZERS_REGISTRY = {
    'coberta': AutoTokenizer,
    'phobert': AutoTokenizer, 
    'vinai/phobert-base': AutoTokenizer,
    'vinai/phobert-large': AutoTokenizer,
    'bert': BertTokenizer,
}

MODEL_PATH_MAP = {
    'coberta': '',
    'phobert': 'vinai/phobert-base',
    'bert': 'bert-base-uncased',
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

def compute_metrics(intent_preds, intent_labels, tag_preds, tag_labels):
    assert len(intent_preds) == len(intent_labels) == len(tag_preds) == len(tag_labels)
    
    results = {}

    intent_result = get_intent_metric(intent_preds, intent_labels)
    tag_result = get_tag_metrics(tag_preds, tag_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, tag_preds, tag_labels)

    results.update(intent_result)
    results.update(tag_result)
    results.update(sementic_result)

    return results

def get_tag_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "tag_precision": seqeval_precision(labels, preds),
        "tag_recall": seqeval_recall(labels, preds),
        "tag_f1": seqeval_f1(labels, preds),
        "tag_report": seqeval_report(labels, preds)
    }

def get_intent_metric(y_true, y_pred):
    """Function to get metrics evaluation.
    
    :param y_pred: Ground truth (correct) target values.
    :param y_true: Estimated targets as returned by a classifier.
    
    :returns: acc, f1, precision, recall
    """

    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred, average="weighted")
    precision = precision_score(y_true, y_pred, average="weighted")
    recall    = recall_score(y_true, y_pred,  average="weighted")
    report    = classification_report(y_true, y_pred)

    results = {
        "intent_acc": acc,
        "intent_f1": f1,
        "intent_precision": precision,
        "intent_recall": recall,
        "intent_report": report
    }

    return results

def get_sentence_frame_acc(intent_preds, intent_labels, tag_preds, tag_labels):
    """For the cases that intent and all the tags are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the tag comparision result
    tag_result = []
    for preds, labels in zip(tag_preds, tag_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        tag_result.append(one_sent_result)

    tag_result = np.array(tag_result)

    sementic_acc = np.multiply(intent_result, tag_result).mean()
    return {
        "mean_acc_score": sementic_acc
    }
