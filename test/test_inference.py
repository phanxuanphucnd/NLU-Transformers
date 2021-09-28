# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

from datetime import datetime

from arizona.nlu.datasets import JointNLUDataset
from arizona.nlu.learners.joint import JointCoBERTaLearner


def test_infer():

    now = datetime.now()

    text = 'Xe bảo hành 1 năm nếu lỗi nhà sx ah b'
    model_path = 'models/phobert-nlu'

    learner = JointCoBERTaLearner(model_name_or_path='phobert')
    learner.load_model(model_path)
    output = learner.predict(
        sample=text,
        lowercase=True,
        rm_emoji=True,
        rm_url=True,
        rm_special_token=True
    )

    rasa_format_output = learner.process(
        sample=text,
        lowcase=True,
        rm_emoji=True,
        rm_url=True,
        rm_special_token=True
    )

    from pprint import pprint
    print("\n>>>>> Output function predict(): ")
    pprint(output)

    print(f"Training time: {datetime.now() - now}")

    print("\n>>>>> Output function process(): ")
    pprint(rasa_format_output)

test_infer()