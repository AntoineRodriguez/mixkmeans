"""
temporary script
"""
import pandas as pd

df = pd.read_csv('./data/data_preprocess/stats_answers.csv')

import pickle

import matplotlib.pyplot as plt

from statistics import quantiles

from collections import Counter

from scripts import SubForum, get_all_words

# android
with open('./data/data_preprocess/android_words.pkl', 'rb') as file:
    words_android = pickle.load(file)
with open('./data/data_preprocess/android.pkl', 'rb') as file:
    android = pickle.load(file)

# VOCAB
len(words_android)
dic = Counter(words_android)
"""
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quantiles(dic.values(), n=10)  # n=100
"""
dic = {k: v for k, v in dic.items() if 181 > v >= 14}

# construction de la matrice document termes
from scipy.sparse import csr_matrix
import numpy as np

# NEEDS
ids = android.questions.index
vocab = list(dic.keys())
# vocab_sorter = np.argsort(vocab) # indices des mots pour les trier


# associer body et title

row_ind = []
col_ind = []
data = []
data_tf = []
dic_idf = dict.fromkeys(vocab, 0)
for row, (doc, content) in enumerate(android.questions['body'].items()):

    for term, nb in Counter(content).items():
        if term in vocab:
            data.append(nb)
            data_tf.append(nb/len(content))  ### attention vocab
            dic_idf[term] += 1  #ne marche pas
            row_ind.append(row)
            col_ind.append()  # comment ??

from sklearn.preprocessing import OneHotEncoder

df.index.apply(lambda str: 'a_' + str)

'''
import numpy as np
data = np.array(['a125214','a3221354','g211354'])
data = ['a125214','a3221354','g211354']
data = [item[0] for item in data]
data = np.array(data)
onehot_encoder = OneHotEncoder(sparse=True)
onehot_encoded = onehot_encoder.fit_transform(data)
onehot_encoded = onehot_encoder.fit_transform(data.reshape(len(data), 1))'''

# DTM =

if __name__ == '__main__':
    pass

