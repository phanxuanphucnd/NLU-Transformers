# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import ast
import requests
import pandas as pd
import streamlit as st

from figures import *
from annotated_text import annotated_text

SERVING_MODEL =  'http://0.0.0.0:8080/api/'

st.title("Nature Language Understanding - NLU")
st.write("*From: phanxuanphucnd@gmail.com*")
st.write('\n\n')

architectures = [
    'OneNet', 
    'JointCoBerta + PhoBERT', 
    'JointCoBerta + PhoBERT + hard-intent_context_embedding', 
    'JointCoBerta + PhoBERT + soft-intent_context_embedding', 
    'JointCoBerta + CoBERTa', 
    'JointCoBerta + CoBERTa + hard-intent_context_embedding',
    'JointCoBerta + CoBERTa + soft-intent_context_embedding',
]
arch_mode = st.selectbox("Choose the architecture mode: ", architectures)

review = st.text_input("Enter text: ")
st.write("\n")
placeholder_btn_pred = st.button('Prediction')
st.write("Press the `Prediction` button...")


################# IC ####################

time = None
text = None
entities = None
intent_ranking = None
if placeholder_btn_pred:
    output = requests.post(url=SERVING_MODEL, json={"text": review, "model_type": arch_mode.lower()})
    output = ast.literal_eval(output.text)
    
    time = output['time']
    prediction = ast.literal_eval(output['output'])
    text = prediction['text']

    intent = prediction['intent'].get('name')
    intent_ranking = prediction['intent'].get('intent_ranking')

    st.subheader("Intent Classification (IC)")
    st.write("\n")
    st.success(intent)

    entities = prediction['entites']

################# INTENT RANKING ####################

if intent_ranking:
    title = 'Intent ranking:'
    intent_list = [i['name'] for i in intent_ranking]
    conf_list = [float(c['softmax_score']) for c in intent_ranking]
    data = pd.DataFrame(list(zip(intent_list, conf_list)), columns=['Intent', 'Confidence'])

    chart = render_most_similar(data, title)
    st.altair_chart(chart)

################# NER ####################

_COLOR_LIST_ = [
    '#8ef', '#faa', '#afa', '#fea', '#8ef', '#7FFFD4', '#C1CDCD', '#E3CF57', '#F5F5DC', '#FFE4C4', '#FFEBCD', '#98F5FF',
    '#EEE8CD', '#B8860B', '#B8860B', '#BF3EFF', '#E9967A', '#C1FFC1', '#00CED1', '#00BFFF', '#00C957', '#FFD700', 
    '#7F7F7F', '#00FF00', '#FF69B4',

]
    

if text and entities:
    st.subheader("Named Entities Recognition (NER)")
    st.write("\n")

    text = text.split()
    assert len(text) == len(entities), f"The length of text different to the length of entities."
    
    list_entities = list(set(entities))
    list_entities.remove('O')
    list_entities = [e[2:] for e in list_entities]

    colors = {}
    for i in range(len(list_entities)):
        colors[list_entities[i]] = _COLOR_LIST_[int(i % len(_COLOR_LIST_))]
    
    notes = []
    for i in range(len(text)):
        if entities[i] == 'O':
            notes.append(" " + text[i] + " ")
        else:
            notes.append((text[i], entities[i], colors[entities[i][2:]]))

    annotated_text(*notes)

if time:
    st.write("\n\n")
    st.success(f"Inference time:   {time}")
