"""
    - Collect data from CQADuspStack and convert it in pandas DataFrame
    - preprocessing
"""
import json

import pandas as pd

from scripts.query_cqadupstack import *
# TODO: optimiser les imports dans ce query_cqadupstack

##############
# FAIRE PLUTOT UN TRUC QUI LIT LES JSON
##############


def json_to_pandas(path_json):
    with open(path_json, 'r') as file:
        temp = json.load(file)
    return pd.DataFrame.from_dict(temp, orient='index')


def get_raw_dataframe(zip_path, save_path):
    """
    Transform zipfiles from CQADupStack datasets into csv,

    zip_path : path to original data, str
    save_path : save "path" 'folder(s)/subforumname', str
    """
    subforum = load_subforum(zip_path)  # create a SubForum object
    posts = pd.DataFrame.from_dict(subforum.postdict, orient='index')
    answers = pd.DataFrame.from_dict(subforum.answerdict, orient='index')
    posts.to_csv(save_path + "_questions.csv")
    answers.to_csv(save_path + "_answers.csv")


if __name__ == "__main__":

    # Collect data -------------------------------
    NAMES = ['android', 'gis', 'physics', 'stats']
    # creating dataframes
    for name in NAMES:
        get_raw_dataframe('../data/{}.zip'.format(name),
                          '../data/data_csv/{}'.format(name))

    # Preprocessing
