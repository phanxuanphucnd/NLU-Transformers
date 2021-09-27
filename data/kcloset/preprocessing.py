import numpy as np
import pandas as pd
import json
from ast import literal_eval
import os
from argparse import ArgumentParser

import pandas as pd
import torch
import numpy as np
from collections import defaultdict
import json


def save_df_as_md(df, out_file, text_col='text', intent_col='intent', tags_col='tags'):
    intents = defaultdict(list)
    for _, row in df.iterrows():
        intent = row[intent_col]
        tokens = row[text_col].split(' ')
        tags = row[tags_col].split(' ')

        s = 0
        e = 0
        entity = None
        text = []
        for i, tag in enumerate(tags):
            if tag[0] == 'B':
                entity = tag[2:]
                s = i
                e = i
                if i == len(tags) - 1:
                    text.append('[{}]({})'.format(' '.join(tokens[s:e + 1]), entity))
            elif tag[0] == 'I':
                e += 1
                if i == len(tags) - 1:
                    text.append('[{}]({})'.format(' '.join(tokens[s:e + 1]), entity))
            elif tag == 'O':
                if entity is not None:
                    text.append('[{}]({})'.format(' '.join(tokens[s:e + 1]), entity))
                    entity = None
                text.append(tokens[i])
        text = ' '.join(text)
        intents[intent].append(text)

    print(out_file)
    with open(out_file, 'w') as pf:
        for intent in intents:
            pf.write('## intent:{}\n'.format(intent))
            for s in intents[intent]:
                pf.write('- {}\n'.format(s))
            pf.write('\n')


def convert_to_csv(examples, out_path: str = "lu_data.csv"):
    """ """
    examples = examples['rasa_nlu_data']['common_examples']

    final_data = []
    for example in examples:
        intent = example['intent']

        entities = example.get('entities', None)

        text = example['text']

        final_data.append({"text": text, "intent": intent,
                           "tags": convert_to_ner(entities, text)})

    data_df = pd.DataFrame(final_data)
    data_df = pd.DataFrame({
        'text': data_df.text,
        'intent': data_df.intent,
        'tags': data_df.tags
    })
    data_df.to_csv(out_path, index=False, encoding='utf-8-sig')

def convert_json_to_csv(input_path, out_path):
    with open(input_path, 'r', encoding='utf-8') as pf:
        data = json.load(pf)
    convert_to_csv(data, out_path)


def convert_to_ner(entities, text):
    list_text_label = []
    tokens = text.split(" ")

    for i in range(len(tokens)):
        list_text_label.append('O')

    if entities is None:
        return ' '.join(list_text_label)

    for info in entities:
        if info["entity"] == 'attribute' or info["entity"] == 'object_type':
            label = '{}:{}'.format(info["entity"], info["value"])
        else:
            label = info["entity"]

        start = info['start']
        end = info['end']

        value = text[start:end]
        list_value = value.split(" ")

        index = len(text[:start].split(" ")) - 1
        list_text_label[index] = 'B-' + str(label)
        for j in range(1, len(list_value)):
            try:
                list_text_label[index + j] = 'I-' + str(label)
            except Exception as e:
                print(str(e))
                print(text)
                print(entities)
    return ' '.join(list_text_label)

parser = ArgumentParser()
parser.add_argument('--input', type=str, default='./train.json')
parser.add_argument('--output', type=str, default='./train.csv')
args = parser.parse_args()


input_path = args.input
output_path = args.output

print(input_path)
print(output_path)
convert_json_to_csv(input_path, output_path)
