# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import random
import numpy as np

from typing import Any
from transformers import AutoTokenizer, RobertaConfig
from seqeval.metrics import precision_score, recall_score, f1_score

from arizona.nlu.models import JointCoBERTa

CONFIGS_REGISTRY = {
    'coberta': RobertaConfig, 
    'phobert': RobertaConfig, 
    'vinai/phobert-base': RobertaConfig,
    'vinai/phobert-large': RobertaConfig,
}

MODELS_REGISTRY = {
    'coberta': JointCoBERTa,
    'phobert': JointCoBERTa, 
    'vinai/phobert-base': JointCoBERTa,
    'vinai/phobert-large': JointCoBERTa,
}

TOKENIZERS_REGISTRY = {
    'coberta': AutoTokenizer,
    'phobert': AutoTokenizer, 
    'vinai/phobert-base': AutoTokenizer,
    'vinai/phobert-large': AutoTokenizer,
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

def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results


def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }


def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }
