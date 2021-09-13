# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch
import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from denver.nlu.modules.crf_module import CRF
from denver.nlu.modules.decoder_module import IntentClassifier, SlotClassifier

class JointCoBERTa(RobertaPreTrainedModel):
    def __init__(
        self, 
        config: PretrainedConfig, 
        dropout: float, 
        intent_label_list: list, 
        slot_label_list: list, 
        use_crf: bool=True, 
        ignore_index: int=0, 
        slot_loss_coef: float=1.0, 
        **kwargs
    ):
        super(JointCoBERTa, self).__init__(config)

        # self.args = args
        self.num_intent_labels = len(intent_label_list)
        self.num_slot_labels = len(slot_label_list)
        self.coberta = RobertaModel(config)
        self.dropout = dropout
        self.use_crf = use_crf
        self.ignore_index = ignore_index
        self.slot_loss_coef = slot_loss_coef

        self.intent_classifier = IntentClassifier(
            input_dim=config.hidden_size, 
            num_intent_labels=self.num_intent_labels, 
            dropout=dropout
        )
        self.slot_classifier = SlotClassifier(
            input_dim=config.hidden_size,
            num_slot_labels=self.num_slot_labels,
            dropout=dropout
        )

        if use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, slot_labels_ids):
        outputs = self.coberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        ) # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]             # [CLS] token

        intent_logits = self.intent_classifier(pooled_output)
        slot_logits = self.slot_classifier(sequence_output)

        total_loss = 0

        # TODO: Intent softmax
        if slot_labels_ids:
            if self.num_intent_labels == 1:
                intent_loss_func = nn.MSELoss()
                intent_loss = intent_loss_func(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_func = nn.CrossEntropyLoss()
                intent_loss = intent_loss_func(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )

            total_loss += intent_loss
        
        # TODO: Slot softmax
        if slot_labels_ids:
            if self.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss          # Negative log-likelihood
            else:
                slot_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                # TODO: Only keep active parts of the loss
                if attention_mask:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_func(active_logits, active_labels)
                else:
                    slot_loss = slot_loss_func(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))

            total_loss += self.slot_loss_coef * slot_loss

        outputs = ((intent_logits, slot_logits), ) + outputs[2:] # add hidden states and attention if they are here
        outputs = (total_loss, ) + outputs                       # (loss), logits, (hidden_states), (attentions)
        
        return outputs 