import os
import random
import numpy as np
import sys

from dataset import Dataset, SpikingDataset, RegSpikingDataset
from torch.utils.data.dataloader import DataLoader

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision
random.seed(1338)
import scipy.io


def pre_processing(train_input, train_label):
    idx_p = 10
    begin = 0 # N_total_frame * N_sync_frame
    
    # label index for data
    train_input_df = pd.DataFrame(train_input, columns = ['1','2', '3', '4'])
    #train_input_df['L1_idx'] = train_input_df.index % idx_p

    # label index for label
    train_label_df = pd.DataFrame(train_label, columns = ['L1','L2'])
    train_label_df['L1_idx'] = train_label_df.index % idx_p
    
    # split training and testing data
    test_input_df, test_label_df = train_input_df.iloc[75* 80 + 1:, :], train_label_df.iloc[75* 80 + 1:, :]
    train_input_df, train_label_df = train_input_df.iloc[:75* 80 + 1, :], train_label_df.iloc[:75* 80 + 1, :]

    # group by 
    #mapping = train_label_df.loc[begin:, :].groupby(by='L1_idx').mean().reset_index().loc[:, ['L1', 'L2', 'L1_idx']] 
    
    #train_input_df = pd.merge(train_input_df, mapping, how='left', on='L1_idx')

    #train_input_df = pd.get_dummies(train_input_df, prefix=['L'], columns=['L1_idx'])

    train_input_df = train_input_df.loc[begin:, :]


    print(train_input_df.head())
    
    # testing data
    # group by
    #test_input_df = test_input_df.merge(mapping, how = 'left', on='L1_idx')

    #test_input_df = pd.get_dummies(test_input_df, prefix=['L'], columns=['L1_idx'])

    print(test_input_df.head())

    train_input = train_input_df.to_numpy()
    test_input = test_input_df.to_numpy()
    
    train_label = train_label_df.drop(['L1_idx'], axis=1).to_numpy()
    test_label = test_label_df.drop(['L1_idx'], axis=1).to_numpy()
    
    print(train_input.shape)
    print(test_input.shape)
    
    
    return train_input, train_label, test_input, test_label


