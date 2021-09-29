# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import logging
import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from arizona.nlu.modules.crf_module import CRF
from arizona.nlu.modules.decoder_module import IntentClassifier, SequenceTaggerClassifier

logger = logging.getLogger(__name__)

class JointCoBERTa(RobertaPreTrainedModel):
    def __init__(
        self, 
        config: PretrainedConfig, 
        dropout: float, 
        intent_labels: list, 
        tag_labels: list, 
        use_crf: bool=True, 
        use_intent_context_concat: bool=False,
        use_intent_context_attention: bool=True,
        attention_embedding_dim: int=200, 
        ignore_index: int=0, 
        max_seq_len: int=50,
        intent_embedding_type: str='soft',
        use_attention_mask: bool=False,
        intent_loss_coef: float=1.0,
        tag_loss_coef: float=1.0
    ):
        super(JointCoBERTa, self).__init__(config)

        self.num_intent_labels = len(intent_labels)
        self.num_tag_labels = len(tag_labels)
        self.roberta = RobertaModel(config)
        self.dropout = dropout
        self.use_crf = use_crf
        self.use_intent_context_concat = use_intent_context_concat
        self.use_intent_context_attention = use_intent_context_attention
        self.attention_embedding_dim = attention_embedding_dim
        self.ignore_index = ignore_index
        self.max_seq_len = max_seq_len
        self.use_attention_mask = use_attention_mask
        self.intent_loss_coef = intent_loss_coef
        self.tag_loss_coef = tag_loss_coef

        if intent_embedding_type not in ['soft', 'hard']:
            logger.warning(
                f"The `embedding_intent_type` must be in ['soft', 'hard']."
                f"Setup the default: `embedding_intent_type='soft'`."
            )
            self.intent_embedding_type = 'soft'
        else:
            self.intent_embedding_type = intent_embedding_type

        self.intent_classifier = IntentClassifier(
            input_dim=config.hidden_size, 
            num_intent_labels=self.num_intent_labels, 
            dropout=dropout
        )
        self.tag_classifier = SequenceTaggerClassifier(
            input_dim=config.hidden_size, 
            num_tag_labels=self.num_tag_labels, 
            num_intent_labels=self.num_intent_labels, 
            use_intent_context_concat=self.use_intent_context_concat, 
            use_intent_context_attention=self.use_intent_context_attention, 
            max_seq_len=self.max_seq_len, 
            attention_embedding_dim=self.attention_embedding_dim, 
            dropout=dropout
        )

        if use_crf:
            self.crf = CRF(num_tags=self.num_tag_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, tag_labels_ids):
        outputs = self.roberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        ) # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]             # [CLS] token

        intent_logits = self.intent_classifier(pooled_output)

        if not self.use_attention_mask:
            tmp_attention_mask = None
        else:
            tmp_attention_mask = attention_mask
        
        if self.intent_embedding_type.lower() == 'hard':
            hard_intent_logits = torch.zeros(intent_logits.shape)
            for i, sample in enumerate(intent_logits):
                max_idx = torch.argmax(sample)
                hard_intent_logits[i][max_idx] = 1
            
            hard_intent_logits = hard_intent_logits.to(self.device)
            tag_logits = self.tag_classifier(sequence_output, hard_intent_logits, tmp_attention_mask)
        else:
            intent_logits = intent_logits.to(self.device)
            tag_logits = self.tag_classifier(sequence_output, intent_logits, tmp_attention_mask)

        total_loss = 0

        # TODO: Intent softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_func = nn.MSELoss()
                intent_loss = intent_loss_func(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_func = nn.CrossEntropyLoss()
                intent_loss = intent_loss_func(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )

            total_loss += self.intent_loss_coef * intent_loss
        
        # TODO: tag softmax
        if tag_labels_ids is not None:
            if self.use_crf:
                tag_loss = self.crf(tag_logits, tag_labels_ids, mask=attention_mask.byte(), reduction='mean')
                tag_loss = -1 * tag_loss          # Negative log-likelihood
            else:
                tag_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                # TODO: Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = tag_logits.view(-1, self.num_tag_labels)[active_loss]
                    active_labels = tag_labels_ids.view(-1)[active_loss]
                    tag_loss = tag_loss_func(active_logits, active_labels)
                else:
                    tag_loss = tag_loss_func(tag_logits.view(-1, self.num_tag_labels), tag_labels_ids.view(-1))

            total_loss += self.tag_loss_coef * tag_loss

        outputs = ((intent_logits, tag_logits), ) + outputs[2:] # add hidden states and attention if they are here
        outputs = (total_loss, ) + outputs                      # (loss), logits, (hidden_states), (attentions)
        
        return outputs # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits