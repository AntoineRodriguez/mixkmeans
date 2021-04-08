import json
import re

import pandas as pd
import nltk

from scripts.stopwords import STOPWORDS

# nltk.download('punkt')


def json_to_pandas(path_json):
    with open(path_json, 'r') as file:
        temp = json.load(file)
    return pd.DataFrame.from_dict(temp, orient='index')


def remove_punctuation(string):
    """Delete punctuation of a sentence"""
    return re.sub(r'[^\w\s]', '', string)


class SubForum:
    def __init__(self, questions_json, answers_json):
        self.questions = json_to_pandas(questions_json)
        self.answers = json_to_pandas(answers_json)
        self.stopwords = STOPWORDS

    def tokenize(self):
        """ Tokenize and remove punctuations in questions and answers bodies """  # noqa
        self.questions['title'] = self.questions.apply(
            lambda row: nltk.word_tokenize(remove_punctuation(row['title'])),
            axis=1)
        self.answers['body'] = self.answers.apply(
            lambda row: nltk.word_tokenize(remove_punctuation(row['body'])),
            axis=1)
        self.questions['body'] = self.questions.apply(
            lambda row: nltk.word_tokenize(remove_punctuation(row['body'])),
            axis=1)

    def autresuppresiondecolonnes(self):
        pass


class SubForumStats(SubForum):
    def delete_columns(self):
        """Delete unwanted columns"""  # pour les statistiques
        self.questions = self.questions[['body', 'title', 'score',
                                         'answers', 'dups']]
        self.answers = self.answers[['body', 'parentid', 'score']]

    def count_answers(self):
        self.questions['nb_answers'] = self.questions.apply(
            lambda row: len(row['answers']),
            axis=1)

    def count_words_questions(self):
        self.questions['nb_words'] = self.questions.apply(
            lambda row: len(row['body']),
            axis=1)

    def count_words_threads(self):
        pass

    def count_duplicates(self):
        self.questions['nb_dups'] = self.questions.apply(
            lambda row: len(row['dups']),
            axis=1)


if __name__ == '__main__':
    pass
