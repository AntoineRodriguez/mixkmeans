"""
temporary script
"""
import pandas as pd

import pickle
from statistics import quantiles
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.preprocessing import OneHotEncoder

from scripts import SubForum, get_all_words

with open('./data/data_preprocess/android_test.pkl', 'rb') as file:
    android = pickle.load(file)

words_android = get_all_words(android)

dic = Counter(words_android)
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quant = quantiles(dic.values(), n=100)  # n=100

dic = {k: v for k, v in dic.items() if quant[-1] > v}

vocab = list(dic.keys())
del dic


android.vocabularize(vocab)
# matrice d'occurence
vectorizer = CountVectorizer(vocabulary=vocab)
# list_1 = vectorizer.get_feature_names() # == vocab
Q_count = vectorizer.fit_transform(list(android.questions['body']))
A_count = vectorizer.fit_transform(list(android.answers['body']))

# matrice tf-idf
transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
Q_tfidf = transformer.fit_transform(Q_count)
A_tfidf = transformer.fit_transform(A_count)


# #ONE HOT ENCODING

'''
data = np.array(['a125214','a3221354','g211354'])
data = ['a125214','a3221354','g211354']
data = [item[0] for item in data]'''

# l'idée c'est ça !
data = [item[0] for item in list(android.questions.index)]
onehot_encoder = OneHotEncoder(sparse=True)
# créer deux P différentes
P = onehot_encoder.fit_transform(data)
P = onehot_encoder.fit_transform(data.reshape(len(data), 1))


# # AFC
# matrice thématiques termes
#  - matrice tt questions / answers tfidf / occurence QUATRE
P.transpose().multiply(Q)


android = SubForum('./data/android/android_questions.json',
                   './data/android/android_answers.json')

android.change_ids('a')

if __name__ == '__main__':
    pass

