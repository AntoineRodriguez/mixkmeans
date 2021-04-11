"""
Preprocessing execution of both questions and answers datasets
"""
import time

from scripts import SubForum
    

if __name__ == '__main__':
    
    #android
    start_time = time.time()
    
    android = SubForum('../data/android/android_questions.json',
                       '../data/android/android_answers.json')
    print('done')
    android.delete_columns()
    android._preprocessing()
    
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
    
    stats.questions.to_csv('../data/data_preprocess/stats_questions.csv', index=True, header=True)
    stats.answers.to_csv('../data/data_preprocess/stats_answers.csv', index=True, header=True)
    
    print("--- %s seconds ---" % (time.time() - start_time))
    del stats
