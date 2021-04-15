import json
import re
import itertools

import pandas as pd
import nltk

from scripts.utils import STOPWORDS, CONTRACTED_FORMS, OTHER_STOPWORDS
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

# nltk.download('punkt')


def json_to_pandas(path_json):
    with open(path_json, 'r') as file:
        temp = json.load(file)
    return pd.DataFrame.from_dict(temp, orient='index')

def compare_id(subforum1, subforum2):
    """
    compare between ids of subforums
    """
    l1 = list(subforum1.questions.index)
    l2 = list(subforum2.questions.index)
    id_q = set(l1) & set(l2)
    
    l3 = list(subforum1.answers.index)
    l4 = list(subforum2.answers.index)
    id_a = set(l3) & set(l4)
    
    return len(id_q), len(id_a) #number of ids that are same

def update_id(subforum):
    """
    update ids of question and answers of subforums
    """    
    #update_id_questions
    id_question_new  = []
    id_question = list(range(len(list(subforum.questions.index))))
    
    for i in id_question:
       id_question_new.append('{}_q_'.format('subforum') + str(i))  # à revoir l'affichage de l'id)
       
    list(subforum.questions.index).clear()
    subforum.questions.index = id_question_new
    
    #update_id_answers
    id_answers_new  = []
    id_answers = list(range(len(list(subforum.answers.index))))
    
    for i in id_answers:
       id_answers_new.append('{}_a_'.format('subforum') + str(i))
       
    list(subforum.answers.index).clear()
    subforum.answers.index = id_answers_new
    
    return len(subforum.questions.index), len(subforum.answers.index)
    
    
def get_all_words(subforum):
    """
    description

    subforum : SubForum object
    """
    temp = subforum.questions['body'].tolist()
    words_list = list(itertools.chain(*temp))
    temp = subforum.questions['title'].tolist()
    words_list += list(itertools.chain(*temp))
    temp = subforum.answers['body'].tolist()
    words_list += list(itertools.chain(*temp))
    return words_list


#df = pd.DataFrame({'a':[[1,2],[1]],'b':[[1,2],[3,4]]})


def cleaning(string):
    """Delete punctuation of a sentence"""
    # remove
    string = re.sub(r'<p>', ' ', string)
    string = re.sub(r'</p>', ' ', string)
    string = re.sub(r'\n', ' ', string)

    # remove numbers
    string = re.sub(r'[0-9]+', ' ', string)

    # standard punctuation
    string = re.sub(r'[\.,;:!\?_\-]', ' ', string)
    # anchors
    string = re.sub(r'[\(\)\]\[\]\{\}\\\/\|]', ' ', string)
    # special characters
    string = re.sub(r'[<>+*=%#&]', ' ', string)
    # currencies
    string = re.sub(r'[£$€]', ' ', string)
    # quotations marks
    string = re.sub(r'[`“”"]', ' ', string)
    # remove possessive ' from words ended by s
    string = re.sub(r'([a-z])\' ', r'\1 ', string)
    return string


class SubForum:
    def __init__(self, questions, answers):
        ext = re.findall(r'.+(\.[a-z]+)', questions)[0]  # '../data/data_preprocess/stats_questions.csv'  # noqa
        if ext == '.csv':
            self.questions = pd.read_csv(questions)
            self.answers = pd.read_csv(answers)
        elif ext == '.json':
            self.questions = json_to_pandas(questions)
            self.answers = json_to_pandas(answers)
        else:
             raise TypeError('not a .csv or a .json file')

        self.stopwords = STOPWORDS

        self.include_title = True

    def link_cleaning(self):
        """ Replace links and urls by ~url~ """
        # https://stackoverflow.com/a/6041965/14836114
        reg = re.compile(r'(http|ftp|https)://([\w_-]+(?:(?:\.[\w_-]+)+))([\w.,@?^=%&:\/~+#-]*[\w@?^=%&\/~+#-])?')  # noqa

        self.answers['body'] = self.answers.apply(
            lambda row: re.sub(reg, '~url~', row['body']),
            axis=1)
        self.questions['body'] = self.questions.apply(
            lambda row: re.sub(reg, '~url~', row['body']),
            axis=1)

    def _cleaning(self):
        """ Discard punctuation in the two dataframes and 'lowerize' strings"""
        self.answers['body'] = self.answers.apply(
            lambda row: cleaning(row['body']).lower(),
            axis=1)
        self.questions['body'] = self.questions.apply(
            lambda row: cleaning(row['body']).lower(),
            axis=1)
        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: cleaning(row['title']).lower(),
                axis=1)

    def expand_contractions(self):
        """ expands the contracted forms in a body answer and question"""
        for c in CONTRACTED_FORMS:
            self.answers['body'] = self.answers.apply(
                lambda row: re.sub(c, CONTRACTED_FORMS[c], row['body']),
                axis=1)
            self.questions['body'] = self.questions.apply(
                lambda row: re.sub(c, CONTRACTED_FORMS[c], row['body']),
                axis=1)
            if self.include_title:
                self.questions['title'] = self.questions.apply(
                    lambda row: re.sub(c, CONTRACTED_FORMS[c], row['title']),
                    axis=1)

    def tokenize(self):
        """ Tokenize by splitting strings (previous 
        process must be done"""

        self.answers['body'] = self.answers.apply(
            lambda row: row['body'].split(),
            axis=1)
        self.questions['body'] = self.questions.apply(
            lambda row: row['body'].split(),
            axis=1)
        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: row['title'].split(),
                axis=1)

    def lemmatize(self):
        """ lemmatization of a body after tokenize it.
        (lemmatization is preferred over Stemming because lemmatization 
        does morphological analysis of the words.)
        """
        lemmatizer = WordNetLemmatizer()

        self.answers['body'] = self.answers.apply(
            lambda row: [lemmatizer.lemmatize(item) for item in row['body']],
            axis=1
        )
        self.questions['body'] = self.questions.apply(
            lambda row: [lemmatizer.lemmatize(item) for item in row['body']],
            axis=1
        )
        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: [lemmatizer.lemmatize(item) for item in row['title']],
                axis=1
            )

    def stemming(self):
        """stemming of a body after tokenize it.
        (stemming:  réduire un mot dans sa forme « racine »
        permet notamment de réduire la taille du vocabulaire dans les approches
        de type sac de mots ou Tf-IdF)
        """
        stemmer = SnowballStemmer(language='english')

        self.answers['body'] = self.answers.apply(
            lambda row: [stemmer.stem(item) for item in row['body']],
            axis=1)

        self.questions['body'] = self.questions.apply(
            lambda row: [stemmer.stem(item) for item in row['body']],
            axis=1)

        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: [stemmer.stem(item) for item in row['title']],
                axis=1
            )

    def remove_stopwords(self):
        self.answers['body'] = self.answers.apply(
            lambda row: [item for item in row['body'] if item not in self.stopwords],  # noqa
            axis=1)

        self.questions['body'] = self.questions.apply(
            lambda row: [item for item in row['body'] if item not in self.stopwords],  # noqa
            axis=1)

        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: [item for item in row['title'] if item not in self.stopwords],  # noqa
                axis=1
            )

    def remove_new_stopwords(self, stopwords):
        """ remove other stopwords found with other methods"""
        self.answers['body'] = self.answers.apply(
            lambda row: [item for item in row['body'] if
                         item not in stopwords],  # noqa
            axis=1)

        self.questions['body'] = self.questions.apply(
            lambda row: [item for item in row['body'] if
                         item not in stopwords],  # noqa
            axis=1)

        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: [item for item in row['title'] if
                             item not in stopwords],  # noqa
                axis=1
            )

    def remove_all_stopwords(self):
        self.answers['body'] = self.answers.apply(
            lambda row: [item for item in row['body'] if
                         item not in self.stopwords + OTHER_STOPWORDS],  # noqa
            axis=1
        )
        self.questions['body'] = self.questions.apply(
            lambda row: [item for item in row['body'] if
                         item not in self.stopwords + OTHER_STOPWORDS],  # noqa
            axis=1
        )
        if self.include_title:
            self.questions['title'] = self.questions.apply(
                lambda row: [item for item in row['title'] if
                             item not in self.stopwords + OTHER_STOPWORDS],  # noqa
                axis=1
            )

    def delete_columns(self):
        """Delete unwanted columns"""
        if self.include_title:
            self.questions = self.questions[['body', 'title', 'score']]
        else:
            self.questions = self.questions[['body', 'score']]

        self.answers = self.answers[['body', 'parentid', 'score']]

    def _preprocessing(self):
        self.link_cleaning()
        self._cleaning()
        self.expand_contractions()
        self.tokenize()
        self.remove_stopwords()
        self.lemmatize()


class SubForumStats(SubForum):
    def __init__(self, questions_json, answers_json):
        super().__init__(questions_json, answers_json)

        self.include_title = True

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
        if 'nb_words' not in self.questions.columns:
            self.count_words_questions()

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

    def _preprocessing(self):
        self._cleaning()
        self.expand_contractions()
        self.tokenize()


if __name__ == '__main__':
    print(STOPWORDS)
