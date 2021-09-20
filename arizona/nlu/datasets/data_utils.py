# -*- coding: utf-8 -*-
# Copyright (c) 2021 by Phuc Phan

import re
import logging
import pandas as pd

from typing import Union
from unicodedata import normalize as nl

logger = logging.getLogger(__name__)

def normalize(
    text, 
    rm_emoji: bool=False, 
    rm_url: bool=False, 
    lowercase: bool=False, 
    rm_special_token: bool=False
):
    '''Function to normalize text
    
    :param text: The text to normalize
    :param lowercase: If True, lowercase data
    :param rm_emoji: If True, replace the emoji token into <space> (" ")
    :param rm_url: If True, replace the url token into <space> (" ")
    :param rm_special_token: If True, replace the special token into <space> (" ")

    :returns: txt: The text after normalize.        
    '''

    # Convert input to UNICODE utf-8
    try:
        txt = nl('NFKC', text)
        # lowercase
        if lowercase:
            txt = txt.lower().strip()
            
        # Remove emoji
        if rm_emoji:
            emoji_pattern = re.compile("["
                                u"\U0001F600-\U0001F64F"  # emoticons
                                u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                                u"\U0001F680-\U0001F6FF"  # transport & map symbols
                                u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                                u"\U00002500-\U00002BEF"  # chinese char
                                u"\U00002702-\U000027B0"
                                u"\U00002702-\U000027B0"
                                u"\U000024C2-\U0001F251"
                                u"\U0001f926-\U0001f937"
                                u"\U00010000-\U0010ffff"
                                u"\u2640-\u2642" 
                                u"\u2600-\u2B55"
                                u"\u200d"
                                u"\u23cf"
                                u"\u23e9"
                                u"\u231a"
                                u"\ufe0f"                 # dingbats
                                u"\u3030"
                                u"\u4e00-\u9fff"          # chinese,japan,korean word
                                "]+", flags=re.UNICODE)

            txt = emoji_pattern.sub(r" ", txt) 

        # Remove url, link
        if rm_url:
            url_regex = re.compile(r'\bhttps?://\S+\b')
            txt = url_regex.sub(r" ", txt)

        # Remove special token and duplicate <space> token
        if rm_special_token:
            txt = re.sub(r"[^a-z0-9A-Z*\sÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠẾếàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂưăạảấầẩẫậắằẳẵặẹẻẽềềểỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸýửữựỳỵỷỹ]", " ", txt)
        
        txt = re.sub(r"\s{2,}", " ", txt)

    except Exception as e:
        logger.error(f"  {text}")
        raise ValueError(f"{e}")

    return txt.strip()

def normalize_df(data_df, rm_emoji, rm_url, rm_special_token, lowercase):
    '''Normalize text data frame
    
    :param data_df: A dataframe
    :param lowercase: If True, lowercase data
    :param rm_emoji: If True, replace the emoji token into <space> (" ")
    :param rm_url: If True, replace the url token into <space> (" ")
    :param rm_special_token: If True, replace the special token into <space> (" ")
    
    :returns: A dataframe after normalized.
    '''
    data_df = data_df.dropna()
    data_df["text"] = data_df["text"].apply(lambda x: normalize(x, rm_emoji=rm_emoji, 
                                rm_url=rm_url, rm_special_token=rm_special_token, lowercase=lowercase))

    return data_df

def standardize_df(
    df: pd.DataFrame, 
    text_col: Union[int, str]='text', 
    label_col: Union[int, str]=None,
    intent_col: Union[int, str]=None, 
    tag_col: Union[int, str]=None, 
):
    """Standardize a dataframe following the standardization format.

    :param df: A DataFrame
    :param text_col: The column name of text data
    :param label_col: The column name of label data
    :param intent_col: The column specify the label of intent with jointly task IC and NER
    :param tag_col: The column specify the label of tagging with jointly task IC NER NER
    
    :return: df: A standardized DataFrame
    """
    
    if intent_col and tag_col:
        df = pd.DataFrame({
            'text': df[text_col],
            'intent': df[intent_col],
            'tag': df[tag_col]
        })        
    elif label_col:
        df = pd.DataFrame({
            'label': df[label_col],
            'text': df[text_col]
        })
    else:
        df = pd.DataFrame({
            'text': df[text_col]
        })
    return df

def get_intent_labels(data_df, intent_col: str='intent', special_intents: list=[]):
    intent_label_list = list(data_df[intent_col])
    intent_labels = list(set(intent_label_list))
    if special_intents:
        special_intents.extend(intent_labels)
        return special_intents
    
    return intent_labels


def get_tag_labels(data_df, tag_col: str='tag', special_tags: list=[]):
    tag_label_list = []
    for i in range(len(data_df)):
        tag_label_list.extend(data_df[tag_col][i].split())
        
    tag_labels = list(set(tag_label_list))
    if special_tags:
        special_tags.extend(tag_labels)
        return special_tags

    return tag_labels


def storages_labels(intent_labels, tag_labels):
    ipath = './intent_labels.txt'
    tpath = './intent_labels.txt'

    with open(ipath, 'w', encoding='utf-8') as f:
        for intent in intent_labels:
            f.writelines(intent + '\n')

    with open(tpath, 'w', encoding='utf-8') as f:
        for tag in tag_labels:
            f.writelines(tag + '\n')