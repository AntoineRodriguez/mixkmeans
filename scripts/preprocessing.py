import json
import re
import pandas as pd
import nltk

from scripts.utils import STOPWORDS, contracted_forms
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer

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

    def lemmatize(self):
        """ lemmatization of a body after tokenize it.
        (lemmatization is preferred over Stemming because lemmatization 
        does morphological analysis of the words.)
        """
        lemmatizer = WordNetLemmatizer()
        
        self.answers['body'] = self.answers.apply(
            lambda row: lemmatizer.lemmatize(nltk.word_tokenize(
                    remove_punctuation(row['body'])), axis=1))
            
        self.questions['body'] = self.questions.apply(
            lambda row: lemmatizer.lemmatize(nltk.word_tokenize(
                    remove_punctuation(row['body'])), axis=1))
        
    def stemming(self):
        """stemming of a body after tokenize it.
        (stemming:  réduire un mot dans sa forme « racine »
        permet notamment de réduire la taille du vocabulaire dans les approches
        de type sac de mots ou Tf-IdF)
        """
        stemmer = SnowballStemmer(language='english')
        
        self.answers['body'] = self.answers.apply(
            lambda row: stemmer.steme(nltk.word_tokenize(
                    remove_punctuation(row['body'])), axis=1))
            
        self.questions['body'] = self.questions.apply(
            lambda row: stemmer.steme(nltk.word_tokenize(
                    remove_punctuation(row['body'])), axis=1))

    def expand_contractions(self):
        """ expands the contracted forms in a body answer and question"""
        for c in contracted_forms:
            self.questions['body'] = re.sub(c, contracted_forms[c], self.questions['body']) #tokenize?
            self.answers['body'] = re.sub(c, contracted_forms[c], self.answers['body'])
        #return self.questions['body']

    def link_cleaning(self):
        """ """
        

    def remove_stopwords(self):
        # TODO : lire processing article 1
        # stopwords et plus utilisés
        pass

    def delete_columns(self):
        """Delete unwanted columns"""
        self.questions = self.questions[['body', 'title', 'score']]
        self.answers = self.answers[['body', 'parentid', 'score']]

    def preprocessing(self):
        #to launch all functions
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
        self.answers['nb_words'] = self.answers.apply(
            lambda row: len(row['body']),
            axis=1
        )
        self.questions['nb_words_threads'] = self.questions['nb_words'].copy()
        # add number of words in title
        self.questions['nb_words_threads'] += self.questions.apply(
            lambda row: len(row['title']),
            axis=1
        )
        # add number of words in each answers
        self.questions['nb_words_threads'] += self.questions.apply(
            lambda row:  sum(self.answers['nb_words'].loc[row['answers']]),
            axis=1
        )
        del self.answers['nb_words']

    def count_duplicates(self):
        self.questions['nb_dups'] = self.questions.apply(
            lambda row: len(row['dups']),
            axis=1)


if __name__ == '__main__':
    pass
