import unittest

from scripts import SubForum, SubForumStats


class SubForumTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    @unittest.skip
    def test_init(self):
        self.object = SubForum('./questions.json',
                               './answers.json')

    def test_discard_punctuation(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._cleaning()
        # print(self.object.questions['body'][0])

    def test_expand_contractions(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object.expand_contractions()
        # print(self.object.questions['body'][0])

    def test_tokenize(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object.link_cleaning()
        self.object._cleaning()
        self.object.expand_contractions()
        self.object.tokenize()
        # print(self.object.questions['body'][0])

    def test_lemmatize(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object.link_cleaning()
        self.object._cleaning()
        self.object.expand_contractions()
        self.object.tokenize()
        self.object.lemmatize()
        # print(self.object.questions['body'][0])

    def test_preprocessing(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        # print(self.object.questions['body'][0])


class SubForumStatsTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    def test_count_answers(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_answers()
        # print(self.object.questions['nb_answers'])

    def test_count_words_questions(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_words_questions()
        # print(self.object.questions['nb_words'])

    def test_count_words_threads(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_words_threads()
        # print(self.object.questions['nb_words_threads'])

    def test_count_duplicates(self):
        self.object = SubForumStats('./questions.json',
                                    './answers.json')
        self.object._preprocessing()
        self.object.count_duplicates()
        # print(self.object.questions['nb_dups'])


if __name__ == '__main__':
    unittest.main()
