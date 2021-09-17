# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import torch.nn as nn

from transformers.configuration_utils import PretrainedConfig
from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel

from arizona.nlu.modules.crf_module import CRF
from arizona.nlu.modules.decoder_module import IntentClassifier, SequenceTaggerClassifier

class JointCoBERTa(RobertaPreTrainedModel):
    def __init__(
        self, 
        config: PretrainedConfig, 
        dropout: float, 
        intent_label_list: list, 
        tag_label_list: list, 
        use_crf: bool=True, 
        ignore_index: int=0, 
        tag_loss_coef: float=1.0, 
        **kwargs
    ):
        super(JointCoBERTa, self).__init__(config)

        # self.args = args
        self.kwargs = kwargs
        self.num_intent_labels = len(intent_label_list)
        self.num_tag_labels = len(tag_label_list)
        self.coberta = RobertaModel(config)
        self.dropout = dropout
        self.use_crf = use_crf
        self.ignore_index = ignore_index
        self.tag_loss_coef = tag_loss_coef

        self.intent_classifier = IntentClassifier(
            input_dim=config.hidden_size, 
            num_intent_labels=self.num_intent_labels, 
            dropout=dropout
        )
        self.tag_classifier = SequenceTaggerClassifier(
            input_dim=config.hidden_size,
            num_tag_labels=self.num_tag_labels,
            dropout=dropout
        )

        if use_crf:
            self.crf = CRF(num_tags=self.num_tag_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, tag_labels_ids):
        outputs = self.coberta(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids, 
        ) # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]             # [CLS] token

        intent_logits = self.intent_classifier(pooled_output)
        tag_logits = self.tag_classifier(sequence_output)

        total_loss = 0

        # TODO: Intent softmax
        if tag_labels_ids:
            if self.num_intent_labels == 1:
                intent_loss_func = nn.MSELoss()
                intent_loss = intent_loss_func(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                intent_loss_func = nn.CrossEntropyLoss()
                intent_loss = intent_loss_func(
                    intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1)
                )

            total_loss += intent_loss
        
        # TODO: tag softmax
        if tag_labels_ids:
            if self.use_crf:
                tag_loss = self.crf(tag_logits, tag_labels_ids, mask=attention_mask.byte(), reduction='mean')
                tag_loss = -1 * tag_loss          # Negative log-likelihood
            else:
                tag_loss_func = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
                # TODO: Only keep active parts of the loss
                if attention_mask:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = tag_logits.view(-1, self.num_tag_labels)[active_loss]
                    active_labels = tag_labels_ids.view(-1)[active_loss]
                    tag_loss = tag_loss_func(active_logits, active_labels)
                else:
                    tag_loss = tag_loss_func(tag_logits.view(-1, self.num_tag_labels), tag_labels_ids.view(-1))

            total_loss += self.tag_loss_coef * tag_loss

        outputs = ((intent_logits, tag_logits), ) + outputs[2:] # add hidden states and attention if they are here
        outputs = (total_loss, ) + outputs                       # (loss), logits, (hidden_states), (attentions)
        
        return outputs 