"""
    Reproduction of some results of the article :
    'CQADupStack: A Benchmark Data Set for Community
    Question-Answering Research' - Hoogeveen, Doris and Verspoor,
    Karin M. and Baldwin, Timothy


"""
from time import time

import pandas as pd
from scripts import SubForumStats


def reproduce_stats(subforum):
    """ Reproduce statistics from article """
    dic = {'# threads': len(subforum.questions)}

    # average number of answers per question
    subforum.count_answers()
    dic['av. # a per q'] = subforum.questions['nb_answers'].mean()

    # average number of words per question
    subforum.count_words_questions()
    dic['av. # w per q'] = subforum.questions['nb_words'].mean()

    # average number of words per thread
    subforum.count_words_threads()
    dic['av. # w per t'] = subforum.questions['nb_words_threads'].mean()

    # percentage of duplicates
    subforum.count_duplicates()
    temp = len(subforum.questions['nb_dups'][subforum.questions['nb_dups'] != 0]) # noqa
    dic['% dups'] = temp*100/len(subforum.questions)

    # average number of duplicates per question
    dic['av. # d per q'] = subforum.questions['nb_dups'].mean()
    return dic


if __name__ == '__main__':
    t = time()
    android = SubForumStats('../data/android/android_questions.json',
                            '../data/android/android_answers.json')
    android.delete_columns()
    android._preprocessing()
    dic = reproduce_stats(android)
    # init dataframe
    df = pd.DataFrame(dic, index=['android'])
    print('execution android {} s'.format(time()-t))
    del android

    t = time()
    gis = SubForumStats('../data/gis/gis_questions.json',
                        '../data/gis/gis_answers.json')
    gis.delete_columns()
    gis._preprocessing()
    dic = reproduce_stats(gis)
    # update df
    temp = pd.Series(dic, name='gis')
    df = df.append(temp)
    print('execution gis {} s'.format(time() - t))
    del gis

    t = time()
    physics = SubForumStats('../data/physics/physics_questions.json',
                            '../data/physics/physics_answers.json')
    physics.delete_columns()
    physics._preprocessing()
    reproduce_stats(physics)
    # update df
    temp = pd.Series(dic, name='physics')
    df = df.append(temp)
    print('execution physics {} s'.format(time() - t))
    del physics

    t = time()
    stats = SubForumStats('../data/stats/stats_questions.json',
                          '../data/stats/stats_answers.json')
    stats.delete_columns()
    stats._preprocessing()
    reproduce_stats(stats)
    # update df
    temp = pd.Series(dic, name='stats')
    df = df.append(temp)
    print('execution stats {} s'.format(time() - t))
    del stats

    # SAVING
    df.to_csv('statistics_hoogeveen.csv')
