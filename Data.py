'''
Descripttion: 
version: 1.0
Author: Suliang Luo
Date: 2023-07-24 20:37:12
LastEditors: Please set LastEditors
LastEditTime: 2023-10-01 11:48:12
'''
import os
import random
import numpy as np
import pickle
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
root_dir = os.path.dirname(os.path.abspath(__file__))


vacal_chars = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9','-']
char2id = {char:i for i,char in enumerate(vacal_chars,start=1)}
id2char = {i:char for i,char in enumerate(vacal_chars,start=1)}
MAX_FEATURES = len(vacal_chars)
print("The value of max_features:", MAX_FEATURES)


# get dataset
def get_data(dir_path):
    X, Y = pd.DataFrame(dtype=int), pd.Series(dtype=int)
    for root,dirs,files in os.walk(dir_path):
        for file in files:
            file_abs = os.path.join(root,file)
            data = pickle.load(open(file_abs,'rb'))
            X_get, Y_get = data['domain_trans'], data['label']
            X = pd.concat([X,X_get])
            Y = pd.concat([Y,Y_get])
    global max_len
    max_len = max(X.apply(lambda x: len(x.max()), axis=1))
    print("The value of max_len:", max_len)
    X = pad_sequences(X[0].tolist(), maxlen=max_len, value=0.,padding='post')
    X = np.array(X)
    Y = np.array(Y)
    return X, Y



def get_oneData(file_path):
    data = pickle.load(open(file_path,'rb'))
    X_get, Y_get = data['domain_trans'], data['label']
    global max_len
    max_len = max([len(x) for x in X_get])
    print("The value of max_len:", max_len)
    X_get = pad_sequences(X_get, maxlen=max_len, value=0., padding='post')
    X_get = np.array(X_get)
    Y_get = np.array(Y_get)
    return X_get, Y_get