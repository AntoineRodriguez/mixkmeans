import os
import unittest

from scipy import sparse

from scripts.mixkmeans import dist, composite_distance


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.dtm = sparse.load_npz('dtm_test.npz')

    @unittest.skip
    def test_dist(self):
        print(dist(self.dtm[0], self.dtm[42]))

    def test_composite_distance(self):
        print(composite_distance(self.dtm[0], self.dtm[86], -3, (0.2, 0.8)))


if __name__ == '__main__':
    unittest.main()
