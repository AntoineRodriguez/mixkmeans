"""
Preprocessing execution of both questions and answers datasets
"""
import pickle
from collections import Counter
from statistics import quantiles

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

from scripts import SubForum, get_all_words

# TODO : ADD COMMENTS

if __name__ == '__main__':
    subforums = SubForum('../data/original_data/android_questions.json',
                         '../data/original_data/android_answers.json')
    # only android for the moment, needs to do that to deal with memory
    subforums.change_ids('a')
    subforums.pre_processing()

    gis = SubForum('../data/original_data/gis_questions.json',
                   '../data/original_data/gis_answers.json')
    gis.change_ids('g')
    gis.pre_processing()
    subforums = subforums + gis
    del gis

    physics = SubForum('../data/original_data/physics_questions.json',
                       '../data/original_data/physics_answers.json')
    physics.change_ids('p')
    physics.pre_processing()
    subforums = subforums + physics
    del physics

    stats = SubForum('../data/original_data/stats_questions.json',
                     '../data/original_data/stats_answers.json')
    stats.change_ids('s')
    stats.pre_processing()
    subforums = subforums + stats
    del stats
    
    # ----------------------
    # Get vocabulary 
    # ----------------------
    words = get_all_words(subforums)
    
    dic = Counter(words)
    dic = {k: v for k, v in dic.items() if v >= 3}
    quant = quantiles(dic.values(), n=100)
    dic = {k: v for k, v in dic.items() if quant[-1] > v}
    vocab = list(dic.keys())
    del dic

    subforums.vocabularize(vocab)
    with open('../data/data_preprocess/subforums.pkl', 'wb') as file:
        pickle.dump(subforums, file)

    # ----------------------
    # Documents-Terms-Matrix
    # ----------------------
    
    vectorizer = CountVectorizer(vocabulary=vocab)

    Q_count = vectorizer.fit_transform(list(subforums.questions['body']))
    with open('../data/data_preprocess/questions_DTM_occ.pkl', 'wb') as file:
        pickle.dump(Q_count, file)

    A_count = vectorizer.fit_transform(list(subforums.answers['body']))
    with open('../data/data_preprocess/answers_DTM_occ.pkl', 'wb') as file:
        pickle.dump(A_count, file)
    del subforums

    # matrice tf-idf
    transformer = TfidfTransformer(smooth_idf=True, use_idf=True)

    Q_tfidf = transformer.fit_transform(Q_count)
    with open('../data/data_preprocess/questions_DTM_tfidf.pkl', 'wb') as file:
        pickle.dump(Q_tfidf, file)
    del Q_count

    A_tfidf = transformer.fit_transform(A_count)
    with open('../data/data_preprocess/answers_DTM_tfidf.pkl', 'wb') as file:
        pickle.dump(A_tfidf, file)
