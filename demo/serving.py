# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import os
import ast
import uvicorn
import tempfile
import numpy as np

from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
from transformers_nlu.nlu.learners.joint import JointCoBERTaLearner

TEMP_DIR = tempfile.mkdtemp()

app = FastAPI()

model_path_1 = '../models/phobert-ks'

# MODEL: OneNet
learner_0 = None

# MODEL: JointCoBerta + PhoBERT
learner_1 = JointCoBERTaLearner(model_type='phobert')
learner_1.load_model(model_path_1)

# MODEL: JointCoBerta + PhoBERT + hard-intent_context_embedding
learner_2 = None

# MODEL: JointCoBerta + PhoBERT + soft-intent_context_embedding
learner_3 = None

# MODEL: JointCoBerta + CoBERTa
learner_4 = None

# MODEL: JointCoBerta + CoBERTa + hard-intent_context_embedding
learner_5 = None

# MODEL: JointCoBerta + CoBERTa + soft-intent_context_embedding
learner_6 = None

learner_maps = {
    'onenet': learner_0,
    'jointcoberta + phobert': learner_1,
    'jointcoberta + phobert + hard-intent_contet_embedding': learner_2,
    'jointcoberta + phobert + soft-intent_contet_embedding': learner_3,
    'jointcoberta + coberta': learner_4,
    'jointcoberta + coberta + hard-intent_contet_embedding': learner_5,
    'jointcoberta + coberta + soft-intent_contet_embedding': learner_6,

}

class InputText(BaseModel):
    text: str
    model_type: str

@app.post("/api/")
async def inference(input: InputText):
    # TODO: Running
    now = datetime.now()

    output = learner_maps[input.model_type.lower()].predict(
        sample=input.text,
        lowcase=True,
        rm_emoji=True,
        rm_url=True,
        rm_special_token=True
    )

    inference_time = datetime.now() - now
    # print(f"- response: 'output': {str(output)}, 'time': {inference_time}")
    
    return {
        "output": str(
            {
                "text": output['text'], 
                "intent": {
                    "name": output['intent'].get('name'),
                    'score': output['intent'].get('confidence'),
                    'intent_ranking': output['intent'].get('intent_ranking')
                },
                "entites": output['tags']['tags']
            }
        ), 
        "time": str(inference_time)}


@app.get("/")
async def root():
    return {"message": "`transformers_nlu` (v0.0.1) is a subpakage of Denver toolbox."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
