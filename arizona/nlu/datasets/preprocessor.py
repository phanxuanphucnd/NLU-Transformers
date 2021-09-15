# -*- coding: utf-8
# Copyright (c) 2021 by Phuc Phan

import re
import logging
import pandas as pd

from typing import Union
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__) 

class BalanceLearning(object):
    """
    Balance a DataFrame.
    """
    def __init__(self):
        """Initialize a BalanceLearning class. """
        super(BalanceLearning, self).__init__()

    @classmethod
    def subtext_sampling(
        cls, 
        data: Union[str, pd.DataFrame], 
        size_per_class: int=None, 
        label_col: str='label',
        replace_mode: bool=False,
    ):
        """Balancing a Dataframe 

        :param data: A dataframe or a path to the .csv file.
        :param size_per_class: Number of items to sampling.
        :param label_col: The column of a dataframe to sampling data follow it. 
        :param replace_mode: Allow or disallow sampling of the same row more than once. 

        """
        if type(data) == str:
            data_df = pd.read_csv(data, encoding='utf-8').dropna()
        else:
            data_df = data

        y = data_df[label_col]

        if size_per_class is None:
            size_per_class = y.value_counts().min()

        list_df = []
        for label in y.value_counts().index:
            samples = data_df[data_df[label_col] == label]
            
            if size_per_class > len(samples) and replace_mode == False:
                samples = samples.sample(n=len(samples), replace=replace_mode)
            else:
                samples = samples.sample(n=size_per_class, replace=replace_mode)

            list_df.append(samples)
        
        data_df = pd.concat(list_df)

        return data_df

def split_data(
    data: pd.DataFrame, 
    pct: float=0.1, 
    is_stratify: bool=False, 
    label_col: str='intent', 
    seed: int=123, 
    *kwargs
):
    """Function to split data into train and test set follow as the pct value
    
    :param data: A data DataFrame
    :param pct: The ratio to split train/test set
    :param is_stratify: If True, data is split in a stratified fashion, using this as the class labels.

    :returns: train_df: A train DataFrame dataset
    :returns: test_df: A test DataFrame dataset
    """
    data = data.dropna()
    if is_stratify:
        train_df, test_df = train_test_split(
            data, test_size=pct, stratify=data[label_col], random_state=seed)
    else:
        train_df, test_df = train_test_split(
            data, test_size=pct, random_state=seed)

    return train_df, test_df
