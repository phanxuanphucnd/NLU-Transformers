# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import enum
import os
import copy
import json
import torch
import logging
import pandas as pd

from pandas import DataFrame
from torch.utils.data import TensorDataset

from arizona.nlu.datasets.preprocessor import BalanceLearning
from arizona.nlu.datasets.preprocessor import standardize_df, normalize_df
from arizona.nlu.datasets.data_utils import get_intent_labels, get_tag_labels

logger = logging.getLogger(__name__)

class InputExample(object):
    """A single training/ test example for simple sequence classification.

    Args:

        guid: Unique id for the example.
        words: List. The words of the sequence.
        intent_label: (Optinal) string. The intent label of the example.
        tag_labels: (Optional) list. The tag labels of the example.
    """
    def __init__(self, guid, words, intent_label=None, tag_labels=None) -> None:
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.tag_labels = tag_labels

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Seriralizes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        
        return output
    
    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class InputFeatures(object):
    """A single set of features of data."""
    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, tag_labels_ids):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.tag_labels_ids = tag_labels_ids
    
    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)

        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

class JointDataProcessor(object):
    """Processor for the JointCoBERTa dataset."""
    def __init__(
        self, 
        data_df: DataFrame=None, 
        intent_labels: list=None, 
        tag_labels: list=None,
        **kwargs
    ):
        self.data_df = data_df
        self.intent_labels = intent_labels
        self.tag_labels = tag_labels

    @classmethod
    def from_csv(
        cls, 
        data_path: str=None, 
        text_col: str=None, 
        intent_col: str=None, 
        tag_col: str=None, 
        special_intents: list=["UNK"],
        special_tags: list=["PAD", "UNK"],
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        balance_data: bool=False,
        size_per_class: int=None,
        replace_mode: bool=False,
        **kwargs
    ):
        if data_path:
            if not data_path.endswith('.csv'):
                raise ValueError(f"File {data_path} is an invalid file format. Must be .csv format file.")
            
            data_df = pd.read_csv(data_path, encoding='utf-8')
        else:
            raise ValueError(f"The parameter `data_path` must be not None value !")

        return cls.from_df(
            data_df, text_col, intent_col, tag_col, 
            special_intents, special_tags, lowercase, 
            rm_emoji, rm_url, rm_special_token, 
            balance_data, size_per_class, replace_mode
        )
    
    @classmethod
    def from_df(
        cls,
        data_df: DataFrame=None,
        text_col: str='text', 
        intent_col: str='intent', 
        tag_col: str='tag', 
        special_intents: list=["UNK"],
        special_tags: list=["PAD", "UNK"],
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        balance_data: bool=False,
        size_per_class: int=None,
        replace_mode: bool=False,
        **kwargs
    ):
        if data_df:
            if balance_data:
                data_df = BalanceLearning().subtext_sampling(
                    data=data_df,
                    text_col=text_col,
                    size_per_class=size_per_class,
                    label_col=intent_col,
                    replace_mode=replace_mode
                )
            data_df = standardize_df(
                df=data_df, text_col=text_col, 
                intent_col=intent_col, tag_col=tag_col
            )
            data_df = normalize_df(
                data_df, rm_emoji=rm_emoji, rm_url=rm_url, 
                rm_special_token=rm_special_token, lowercase=lowercase
            )

            intent_labels = get_intent_labels(data_df, intent_col, special_intents)
            tag_labels = get_tag_labels(data_df, tag_col, special_tags)
        else:
            raise ValueError(f"The parameter `data_df` must be not None value !")

        return cls(data_df, intent_labels, tag_labels)

    def _create_examples(self, texts, intents, tags, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for i, (text, intent, tag) in enumerate(zip(texts, intents, tags)):
            guid = "%s-%s" % (set_type, i)

            # TODO: Input text
            words = text.split()
            
            # TODO: Intent
            intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")

            # TODO: Tags
            tag_labels = []
            for s in tag.split():
                tag_labels.append(self.tag_labels.index(s) if s in self.tag_labels else self.tag_labels.index("UNK"))

            assert len(words) == len(tag_labels)

            examples.append(InputExample(guid=guid, words=words, intent_label=intent_label, tag_labels=tag_labels))

        return examples

    def get_examples(self, mode):
        """Get all examples from data.
        
        Args:
            mode: train, dev, test.
        """
        logger.info(f"LOOKING AT MODE: {mode}")

        return self._create_examples(texts=list(self.data_df['text']),
                                     intents=list(self.data_df['intent']),
                                     tags=list(self.data_df['tag']),
                                     set_type=mode)