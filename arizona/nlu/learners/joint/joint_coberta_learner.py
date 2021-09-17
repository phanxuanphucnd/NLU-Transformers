# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import torch
import logging
import numpy as np

from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from arizona.utils import set_seed
from arizona.utils import compute_metrics
from arizona.utils import get_from_registry
from arizona.nlu.models.joint import JointCoBERTa
from arizona.utils import CONFIGS_REGISTRY, MODELS_REGISTRY
from arizona.nlu.datasets.joint_dataset import JointNLUDataset

logger = logging.getLogger(__name__)

class JointCoBERTaLearner():
    def __init__(
        self, 
        model: JointCoBERTa=None, 
        model_name_or_path: str=None,
        model_type: str=None,
        device: str=None,
        seed: int=123, 
        **kwargs
    ):
        super(JointCoBERTaLearner, self).__init__()
        
        set_seed(seed)
        
        self.dropout = kwargs.pop('dropout', 0.1)
        self.use_crf = kwargs.pop('use_crf', True)
        self.ignore_index = kwargs.pop('ignore_index', 0)
        self.tag_loss_coef = kwargs.pop('tag_loss_coef', 1.0)
        self.pad_token_label_id = self.ignore_index

        self.model = model
        self.kwargs =kwargs
        self.model_name_or_path = model_name_or_path

        if model_type:
            self.model_type = model_type
        else:
            self.model_type = model_name_or_path

        self.config_class = get_from_registry(model_type, CONFIGS_REGISTRY)
        self.config = self.config_class.from_pretrained(model_name_or_path, finetuning_task='nlu')
        self.model_class = get_from_registry(model_type, MODELS_REGISTRY)

        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

    @property
    def __name__(self):
        return 'JointCoBERTaNLU'

    def train(
        self, 
        train_dataset: JointNLUDataset=None,
        test_dataset: JointNLUDataset=None,
        train_batch_size: int=32,
        eval_batch_size: int=64,
        learning_rate: float=5e-5,
        n_epochs: int=10,
        weight_decay: float=0.0,
        gradient_accumulation_steps: int=1,
        adam_epsilon: float=1e-8,
        max_grad_norm: float=1.0,
        max_steps: int=-1,
        warmup_steps: int=0,
        logging_steps: int=200,
        save_steps: int=1000,
        view_model: bool=True, 
        model_dir: str='./model',
        model_name: str='coberta-mini.nlu',
        **kwargs
    ):
        logger.info(f"Dataset Info")
        logger.info(f"Length of Training dataset: {len(train_dataset)}")
        logger.info(f"Length of Test dataset: {len(test_dataset)}")
        logger.info(f"Description intent classes: {train_dataset.processor.intent_labels}")
        logger.info(f"Description tag classes: {train_dataset.processor.tag_labels}")

        self.intent_label_list = train_dataset.processor.intent_labels
        self.tag_label_list = train_dataset.processor.tag_labels

        if not self.model and not self.model_name_or_path:
            raise ValueError(f"Either parameter `model` or `model_name_or_path` must be not None value !")
        elif not self.model:
            self.model = self.model_class.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                dropout=self.dropout,
                use_crf=self.use_crf,
                ignore_index=self.ignore_index,
                tag_loss_coef=self.tag_loss_coef,
                intent_label_list=self.intent_label_list,
                tag_label_list=self.tag_label_list
            )

        self.model.to(self.device)

        if view_model:
            logger.info(f"Model Info")
            print(self.model)

        train_sampler = RandomSampler(train_dataset)
        train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=train_batch_size)

        if max_steps > 0:
            t_total = max_steps
            n_epochs = max_steps // (len(train_dataloader) // gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // gradient_accumulation_steps * n_epochs

        # TODO: Prepare optimizer and schedule (Linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped__parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped__parameters, lr=learning_rate, eps=adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)

        logger.info(f"➖➖➖➖➖ Running training ➖➖➖➖➖")
        logger.info(f" Num examples = {len(train_dataset)}")
        logger.info(f" Num epochs = {n_epochs}")
        logger.info(f" Total train batch size = {train_batch_size}")
        logger.info(f" Gradient accumulation steps = {gradient_accumulation_steps}")
        logger.info(f" Total optimization steps = {t_total}")
        logger.info(f" Logging steps = {logging_steps}")
        logger.info(f" Save steps = {save_steps}")

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(n_epochs), desc="Epoch")
        
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc='Iteration')
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)

                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2], 
                    'intent_label_ids': batch[3],
                    'tag_labels_ids': batch[4]
                }

                outputs = self.model(**inputs)
                loss = outputs[0]

                if gradient_accumulation_steps > 1:
                    loss = loss / gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)

                    optimizer.step()
                    scheduler.step() # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if logging_steps > 0 and global_step % logging_steps == 0:
                        self.evaluate(test_dataset, eval_batch_size)
                    
                    if save_steps > 0 and global_step % save_steps == 0:
                        self.save_model(model_dir, model_name)

            if 0 < max_steps < global_step:
                epoch_iterator.close()
                break

        return global_step, tr_loss / global_step
    
    def evaluate(self, dataset: JointNLUDataset, batch_size: int=64):
        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=batch_size)

        logger.info(f"➖➖➖➖➖ Running evaluation ➖➖➖➖➖")
        logger.info(f" Num exmaples = {len(dataset)}")
        logger.info(f" Batch size = {batch_size}")

        eval_loss = 0
        nb_eval_steps = 0
        intent_preds = None
        tag_preds = None
        out_intent_label_ids = None
        out_tag_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc='Evaluating'):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'intent_label_ids': batch[3],
                    'tag_labels_ids': batch[4]
                }

                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, tag_logits) = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            
            nb_eval_steps += 1

            # TODO: Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # TODO: Tag prediction
            if tag_preds is None:
                if self.use_crf:
                    tag_preds = np.array(self.model.crf.decode(tag_logits))
                else:
                    tag_preds = tag_logits.detach().cpu().numpy()

                out_tag_labels_ids = inputs["tag_labels_ids"].detach().cpu().numpy()
            else:
                if self.use_crf:
                    tag_preds = np.append(tag_preds, np.array(self.model.crf.decode(tag_logits)), axis=0)
                else:
                    tag_preds = np.append(tag_preds, tag_logits.detach().cpu().numpy(), axis=0)

                out_tag_labels_ids = np.append(out_tag_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)

        
        eval_loss = eval_loss / nb_eval_steps
        results = {
            'loss': eval_loss
        }

        # TODO: Intent results
        intent_preds = np.argmax(intent_preds)

        # TODO: Tag results
        if not self.use_crf:
            tag_preds = np.argmax(tag_preds, axis=2)

        tag_label_map = {i: label for i, label in enumerate(self.tag_label_list)}
        out_tag_label_list = [[] for _ in range(out_tag_labels_ids.shape[0])]
        tag_preds_list = [[] for _ in range(out_tag_labels_ids.shape[0])]

        for i in range(out_tag_labels_ids.shape[0]):
            for j in range(out_tag_labels_ids.shape[1]):
                if out_tag_labels_ids[i, j] != self.pad_token_label_id:
                    out_tag_label_list[i].append(tag_label_map[out_tag_labels_ids[i][j]])
                    tag_preds_list[i].append(tag_label_map[tag_preds[i][j]])

        total_results = compute_metrics(intent_preds, out_intent_label_ids, tag_preds_list, out_tag_label_list)
        results.update(total_results)

        logger.info(f"➖➖➖➖➖ Evaluation results ➖➖➖➖➖")
        for key in sorted(results.keys()):
            logger.info(f" {key} = {str(results[key])}")

        return results


    def predict(self):
        raise NotImplementedError

    def process(self):
        raise NotImplementedError

    def save_model(self, model_dir: str='./models', model_name: str='nlu-coberta-mini'):
        model_path = os.path.join(model_dir, model_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        
        model_to_save = self.model.module if hasattr(self.mode, 'module') else self.model
        model_to_save.save_pretrained(model_path)

        # TODO: Save training arguments togethor with the trained model
        torch.save(
            {
                'dropout': self.dropout,
                'use_crf': self.use_crf,
                'ignore_index': self.ignore_index,
                'tag_loss_coef': self.tag_loss_coef,
                'intent_label_list': self.intent_label_list,
                'tag_label_list': self.tag_label_list,
            },
            os.path.join(model_path, 'training_args.bin')
        )

        logger.info(f"Saving model checkpoint to {model_path}")

    def load_model(self, model_path: str=None):
        if not os.path.exists(model_path):
            raise Exception(f"Model doesn't exists or broken !")
        
        try:
            checkpoint = torch.load(os.path.join(model_path, 'training_args.bin'))
            self.dropout = checkpoint.get('dropout')
            self.use_crf = checkpoint.get('use_crf')
            self.ignore_index = checkpoint.get('ignore_index')
            self.tag_loss_coef = checkpoint.get('tag_loss_coef')
            self.intent_label_list = checkpoint.get('intent_label_list')
            self.tag_label_list = checkpoint.get('tag_label_list')

            self.model = self.model_class.from_pretrained(
                model_path,
                config=self.config,
                dropout=self.dropout,
                use_crf=self.use_crf,
                ignore_index=self.ignore_index,
                tag_loss_coef=self.tag_loss_coef,
                intent_label_list=self.intent_label_list,
                tag_label_list=self.tag_label_list
            )

            self.model.to(self.device)
            logger.info(f"******* Model Loaded *******")

        except:
            raise Exception(f"Some model files might be missing...")