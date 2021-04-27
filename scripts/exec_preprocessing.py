"""
Preprocessing execution of both questions and answers datasets
"""
import pickle
from collections import Counter
from statistics import quantiles
import sys
import os

#os.path.dirname()
#sys.path.append(os.getcwd())
#print(os.path.dirname(os.getcwd()))
#print(sys.path)
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from scipy import sparse

from scripts import SubForum, get_all_words, del_line
#os.chdir((os.getcwd()+'\scripts'))


def reduce_words(words):
    dic = Counter(words)
    dic = {k: v for k, v in dic.items() if v >= 3}
    quant = quantiles(dic.values(), n=100)
    dic = {k: v for k, v in dic.items() if quant[-1] > v}
    return list(dic.keys())


# TODO : ADD COMMENTS

android = SubForum('./data/original_data/android_questions.json',
                     './data/original_data/android_answers.json')
# only android for the moment, needs to do that to deal with memory
android.change_ids('a')
"""
ens_android = q_index inter setparentid
"""
q_index_android = set(android.questions.index)   #len(q_index_android) = 23697 
a_parentid_android = set(android.answers.parentid) #len(a_parentid_android) = 17575
ens_android = list(q_index_android & a_parentid_android)   #len(ens_android) = 17575  # ==> il ya des quest sans rép

del_line(q_index_android,ens_android)

android.pre_processing()

q_index = list(android.questions.index)
a_index = list(android.answers.index)

vocab = reduce_words(get_all_words(android))

with open('./data/data_preprocess/android.pkl', 'wb') as file:
    pickle.dump(android, file)
del android

gis = SubForum('./data/original_data/gis_questions.json',
               './data/original_data/gis_answers.json')
gis.change_ids('g')

q_index_gis = set(gis.questions.index)     #len(q_index_gis) = 38522      
a_parentid_gis = set(gis.answers.parentid)  #len(a_parentid_gis) = 30395
ens_gis = list(q_index_gis & a_parentid_gis)  #len(ens_gis) = 30395      # ==> il y a des quest sans rép

gis.pre_processing()

q_index += list(gis.questions.index)
a_index += list(gis.answers.index)

vocab += reduce_words(get_all_words(gis))

with open('./data/data_preprocess/gis.pkl', 'wb') as file:
    pickle.dump(gis, file)
del gis

physics = SubForum('./data/original_data/physics_questions.json',
                   './data/original_data/physics_answers.json')
physics.change_ids('p')

q_index_physics = set(physics.questions.index)     #len(q_index_physics) = 39355      
a_parentid_physics = set(physics.answers.parentid)  #len(a_parentid_physics) = 33271
ens_physics = list(q_index_physics & a_parentid_physics)  #len(ens_physics) = 33271  # ==> il y a des quest sans rép

physics.pre_processing()

q_index += list(physics.questions.index)
a_index += list(physics.answers.index)

vocab += reduce_words(get_all_words(physics))

with open('./data/data_preprocess/physics.pkl', 'wb') as file:
    pickle.dump(physics, file)
del physics

stats = SubForum('./data/original_data/stats_questions.json',
                 './data/original_data/stats_answers.json')
stats.change_ids('s')

q_index_stats = set(stats.questions.index)     #len(q_index_stats) = 42921      
a_parentid_stats = set(stats.answers.parentid)  #len(a_parentid_stats) = 29006
ens_stats = list(q_index_stats & a_parentid_stats)  #len(ens_stats) = 29006  # ==> il y a des quest sans rép

stats.pre_processing()

q_index += list(stats.questions.index)
a_index += list(stats.answers.index)

vocab += reduce_words(get_all_words(stats))

with open('./data/data_preprocess/stats.pkl', 'wb') as file:
    pickle.dump(stats, file)
del stats

with open('./data/data_preprocess/indexes.pkl', 'wb') as file:
    pickle.dump([a_index, q_index], file)

vocab = list(set(vocab))

with open('./data/data_preprocess/vocab.pkl', 'wb') as file:
    pickle.dump(vocab, file)

# ----------------------
# Documents-Terms-Matrix
# ----------------------

with open('./data/data_preprocess/android.pkl', 'rb') as file:
    android = pickle.load(file)

vectorizer = CountVectorizer(vocabulary=vocab)
Q_count = vectorizer.fit_transform(list(android.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
A_count = vectorizer.fit_transform(list(android.answers['body'].apply(lambda sentence: ' '.join(sentence)))) # noqa

android.remove_body()
with open('./data/data_preprocess/android_.pkl', 'wb') as file:
    pickle.dump(android, file)
del android


with open('./data/data_preprocess/gis.pkl', 'rb') as file:
    gis = pickle.load(file)

vectorizer = CountVectorizer(vocabulary=vocab)
temp = vectorizer.fit_transform(list(gis.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
Q_count = sparse.vstack([Q_count, temp])
temp = vectorizer.fit_transform(list(gis.answers['body'].apply(lambda sentence: ' '.join(sentence)))) # noqa
A_count = sparse.vstack([A_count, temp])
del temp

gis.remove_body()
with open('./data/data_preprocess/gis_.pkl', 'wb') as file:
    pickle.dump(gis, file)
del gis

with open('./data/data_preprocess/physics.pkl', 'rb') as file:
    physics = pickle.load(file)

vectorizer = CountVectorizer(vocabulary=vocab)
temp = vectorizer.fit_transform(list(physics.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
Q_count = sparse.vstack([Q_count, temp])
temp = vectorizer.fit_transform(list(physics.answers['body'].apply(lambda sentence: ' '.join(sentence)))) # noqa
A_count = sparse.vstack([A_count, temp])
del temp

physics.remove_body()
with open('./data/data_preprocess/physics_.pkl', 'wb') as file:
    pickle.dump(physics, file)
del physics

with open('./data/data_preprocess/stats.pkl', 'rb') as file:
    stats = pickle.load(file)

vectorizer = CountVectorizer(vocabulary=vocab)
temp = vectorizer.fit_transform(list(stats.questions['body'].apply(lambda sentence: ' '.join(sentence))))  # noqa
Q_count = sparse.vstack([Q_count, temp])
temp = vectorizer.fit_transform(list(stats.answers['body'].apply(lambda sentence: ' '.join(sentence)))) # noqa
A_count = sparse.vstack([A_count, temp])

stats.remove_body()
with open('./data/data_preprocess/stats_.pkl', 'wb') as file:
    pickle.dump(stats, file)
del stats


with open('./data/data_preprocess/questions_DTM_occ.pkl', 'wb') as file:
    pickle.dump(Q_count, file)
with open('./data/data_preprocess/answers_DTM_occ.pkl', 'wb') as file:
    pickle.dump(A_count, file)


# TF IDF
transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

Q_tfidf = transformer.fit_transform(Q_count)                           # <-- faut il la lecture de Q_count ?
with open('./data/data_preprocess/questions_DTM_tfidf.pkl', 'wb') as file:
    pickle.dump(Q_tfidf, file)
del Q_count

with open('./data/data_preprocess/answers_DTM_occ.pkl', 'rb') as file:
    A_count = pickle.load(file)

A_tfidf = transformer.fit_transform(A_count)
with open('./data/data_preprocess/answers_DTM_tfidf.pkl', 'wb') as file:
    pickle.dump(A_tfidf, file)