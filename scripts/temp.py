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

# stats
with open('./data/data_preprocess/stats_words.pkl', 'rb') as file:
    words_stats = pickle.load(file)

len(words_stats)
dic = Counter(words_stats)
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quantiles(dic.values(), n=10)

# android
with open('./data/data_preprocess/android_words.pkl', 'rb') as file:
    words_android = pickle.load(file)

len(words_android)
dic = Counter(words_android)
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quantiles(dic.values(), n=10)  # n=100

# physics
with open('./data/data_preprocess/physics_words.pkl', 'rb') as file:
    words_physics = pickle.load(file)

len(words_physics)
dic = Counter(words_physics)
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quantiles(dic.values(), n=10)

# gis
with open('./data/data_preprocess/gis_words.pkl', 'rb') as file:
    words_gis = pickle.load(file)

len(words_gis)
dic = Counter(words_gis)
dic = {k: v for k, v in dic.items() if v >= 3}
len(dic)
quantiles(dic.values(), n=10)

if __name__ == '__main__':
    pass

