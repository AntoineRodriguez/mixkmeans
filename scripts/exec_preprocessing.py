"""
Preprocessing execution of both questions and answers datasets
"""
import time
import pickle

from scripts import SubForum, get_all_words
    

if __name__ == '__main__':

    #android
    # start_time = time.time()
    
    android = SubForum('../data/android/android_questions.json',
                       '../data/android/android_answers.json')
    """
    android.delete_columns()
    # regarder les id et comparer
    android._preprocessing()

    words_android = get_all_words(android)
    with open('../data/data_preprocess/android_words.pkl', 'wb') as file:
        pickle.dump(words_android, file)
    del words_android

    with open('../data/data_preprocess/android_subforum.pkl', 'wb') as file:
        pickle.dump(android, file)

        # quand lecture mettre 'rb' et importer subforum
    
    android.questions.to_csv('../data/data_preprocess/android_questions.csv', index=True, header=True)
    android.answers.to_csv('../data/data_preprocess/android_answers.csv', index=True, header=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    del android
    #gis
    start_time = time.time()
    
    gis = SubForum('../data/gis/gis_questions.json',
                   '../data/gis/gis_answers.json')
    
    gis.delete_columns()
    gis._preprocessing()

    words_gis = get_all_words(gis)
    with open('../data/data_preprocess/gis_words.pkl', 'wb') as file:
        pickle.dump(words_gis, file)
    del words_gis

    gis.questions.to_csv('../data/data_preprocess/gis_questions.csv', index=True, header=True)
    gis.answers.to_csv('../data/data_preprocess/gis_answers.csv', index=True, header=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    del gis
    
    #physics
    start_time = time.time()
    
    physics = SubForum('../data/physics/physics_questions.json',
                       '../data/physics/physics_answers.json')
    
    physics.delete_columns()
    physics._preprocessing()

    words_physics = get_all_words(physics)
    with open('../data/data_preprocess/physics_words.pkl', 'wb') as file:
        pickle.dump(words_physics, file)
    del words_physics
    
    physics.questions.to_csv('../data/data_preprocess/physics_questions.csv', index=True, header=True)
    physics.answers.to_csv('../data/data_preprocess/physics_answers.csv', index=True, header=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    del physics

    #stats
    start_time = time.time()
    
    stats = SubForum('../data/stats/stats_questions.json',
                     '../data/stats/stats_answers.json')

    stats.delete_columns()
    stats._preprocessing()
    words_stats = get_all_words(stats)
    with open('../data/data_preprocess/stats_words.pkl', 'wb') as file:
        pickle.dump(words_stats, file)
    del words_stats
    
    stats.questions.to_csv('../data/data_preprocess/stats_questions.csv', index=True, header=True)
    stats.answers.to_csv('../data/data_preprocess/stats_answers.csv', index=True, header=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    del stats
"""

