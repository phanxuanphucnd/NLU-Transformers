# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from numpy.lib.arraysetops import isin
import torch
import logging

from typing import Union
from torch.utils.data import TensorDataset
from transformers import PreTrainedTokenizerBase

from arizona.utils import TOKENIZERS_REGISTRY, MODEL_PATH_MAP, get_from_registry
from arizona.nlu.datasets.joint_processor import JointDataProcessor, InputFeatures

logger = logging.getLogger(__name__)

class JointNLUDataset():
    def __init__(
        self, 
        mode: str='train', 
        data_path: str=None, 
        text_col: str=None, 
        intent_col: str=None, 
        tag_col: str=None, 
        special_intents: list=["UNK"],
        special_tags: list=["PAD", "UNK"],
        max_seq_len: int=100,
        ignore_index: int=0,
        lowercase: bool=True, 
        rm_emoji: bool=True, 
        rm_url: bool=True, 
        rm_special_token: bool=False, 
        balance_data: bool=False,
        size_per_class: int=None,
        replace_mode: bool=False,
        tokenizer: Union[str, PreTrainedTokenizerBase]=None,
        **kwargs
    ):
        self.mode = mode
        self.data_path = data_path
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len

        # TODO: Get tokenizer
        if isinstance(tokenizer, PreTrainedTokenizerBase):
            self.tokenizer = tokenizer
        else:
            self.tokenizer = get_from_registry(
                tokenizer, TOKENIZERS_REGISTRY).from_pretrained(
                    MODEL_PATH_MAP.get(tokenizer, None)
                )
        # TODO: Initialize processor
        self.processor = JointDataProcessor.from_csv(
            data_path=data_path, 
            text_col=text_col,
            intent_col=intent_col,
            tag_col=tag_col,
            special_intents=special_intents,
            special_tags=special_tags,
            lowercase=lowercase,
            rm_emoji=rm_emoji,
            rm_url=rm_url,
            rm_special_token=rm_special_token,
            balance_data=balance_data,
            size_per_class=size_per_class,
            replace_mode=replace_mode
        )

    def __len__(self):
        return len(self.processor.data_df)

    def build_dataset(self):
        logger.info(f"Creates features from dataset file at {self.data_path}")
        examples = self.processor.get_examples(self.mode)

        pad_token_label_id = self.ignore_index
        features = self.convert_examples_to_features(
            examples, self.max_seq_len, self.tokenizer, pad_token_label_id=pad_token_label_id)

        # TODO: Convert to Tensors and build Dataset
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
        all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
        all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)

        dataset = TensorDataset(all_input_ids, all_attention_mask,
                                all_token_type_ids, all_intent_label_ids, all_slot_labels_ids)
        
        return dataset

    def convert_examples_to_features(
        self,
        examples,
        max_seq_len,
        tokenizer,
        pad_token_label_id=-100,
        cls_token_segment_id=0,
        pad_token_segment_id=0,
        sequence_a_segment_id=0,
        mask_padding_with_zero=True
    ):
        # TODO: Setting based on the current model type
        cls_token = tokenizer.cls_token
        sep_token = tokenizer.sep_token
        unk_token = tokenizer.unk_token
        pad_token_id = tokenizer.pad_token_id

        features = []
        for (ex_index, example) in enumerate(examples):
            if ex_index % 5000 == 0:
                logger.info("Writing example %d of %d" % (ex_index, len(examples)))

            # TODO: Tokenize word by word (for NER)
            tokens = []
            tag_labels_ids = []
            for word, tag_label in zip(example.words, example.tag_labels):
                word_tokens = tokenizer.tokenize(word)
                if not word_tokens:
                    word_tokens = [unk_token]  # For handling the bad-encoded word
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                tag_labels_ids.extend([int(tag_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

            # TODO: Account for [CLS] and [SEP]
            special_tokens_count = 2
            if len(tokens) > max_seq_len - special_tokens_count:
                tokens = tokens[:(max_seq_len - special_tokens_count)]
                tag_labels_ids = tag_labels_ids[:(max_seq_len - special_tokens_count)]

            # TODO: Add [SEP] token
            tokens += [sep_token]
            tag_labels_ids += [pad_token_label_id]
            token_type_ids = [sequence_a_segment_id] * len(tokens)

            # TODO: Add [CLS] token
            tokens = [cls_token] + tokens
            tag_labels_ids = [pad_token_label_id] + tag_labels_ids
            token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # TODO: The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # TODO: Zero-pad up to the sequence length.
            padding_length = max_seq_len - len(input_ids)
            input_ids = input_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
            token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
            tag_labels_ids = tag_labels_ids + ([pad_token_label_id] * padding_length)

            assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
            assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
            assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
            assert len(tag_labels_ids) == max_seq_len, "Error with tag labels length {} vs {}".format(len(tag_labels_ids), max_seq_len)

            intent_label_id = int(example.intent_label)

            if ex_index < 5:
                logger.info("*** Example ***")
                logger.info("guid: %s" % example.guid)
                logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
                logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
                logger.info("tag_labels: %s" % " ".join([str(x) for x in tag_labels_ids]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              intent_label_id=intent_label_id,
                              tag_labels_ids=tag_labels_ids
                             ))

        return features
