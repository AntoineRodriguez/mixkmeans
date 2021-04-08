import unittest

from scripts import SubForum, SubForumStats


class SubForumTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    @unittest.skip
    def test_init(self):
        self.object = SubForum('../data/android/android_questions.json',
                               '../data/android/android_answers.json')


class SubForumStatsTest(unittest.TestCase):
    def setUp(self):
        self.object = None

    def test_count_answers(self):
        self.object = SubForumStats('../data/android/android_questions.json',
                                    '../data/android/android_answers.json')
        self.object.delete_columns()
        self.object.tokenize()
        self.object.count_answers()


if __name__ == '__main__':
    unittest.main()
