import os
import unittest

from scipy import sparse
import matplotlib.pyplot as plt

from scripts.mixkmeans import (
    dist_eucl,
    dist_cosin,
    composite_distance,
)

from scripts.mixkmeans import MixKMeans


class FunctionsTest(unittest.TestCase):
    def setUp(self):
        self.dtm = sparse.load_npz('dtm_test.npz')

    @unittest.skip
    def test_dist(self):
        print(dist_eucl(self.dtm[0], self.dtm[42]))

    @unittest.skip
    def test_dist_cosin(self):
        print(dist_cosin(self.dtm[0], self.dtm[42]))

    def test_composite_distance(self):
        print(composite_distance(self.dtm[0], self.dtm[86], -3, (0.2, 0.8)))


class MixKMeansTest(unittest.TestCase):
    def setUp(self):
        self.dtm = sparse.load_npz('dtm_test.npz')
        self.dtm_2 = sparse.load_npz('../data/data_preprocess/dtm_occ.npz')
        self.model = MixKMeans(x=-3, weights=(0.2, 0.8))

    def test_initialize_prototypes(self):
        self.model.initialize_prototypes(self.dtm, 4)
        print(self.model.prototypes)

    def test_assign_clusters(self):
        self.model.initialize_prototypes(self.dtm, 4)
        assignation = self.model.assign_clusters(self.dtm)
        print(assignation)

    # REVOIR AVEC UN NOUVEAU DATASET DE TEST!
    def test_compute_prototypes(self):
        self.model.initialize_prototypes(self.dtm, 4)
        assignation = self.model.assign_clusters(self.dtm)
        self.model.compute_prototypes(self.dtm, assignation)
        #print(self.model.prototypes)

    @unittest.skip
    def test_fit(self):
        _, __, cost = self.model.fit(self.dtm, 4, 30)
        print(cost)

    def test_fit_watch(self):
        _, __, cost = self.model.fit(self.dtm, 4, 30)
        print(cost)
        plt.plot(self.model.cost_historic)
        plt.savefig('cost_historic_test.png')


if __name__ == '__main__':
    unittest.main()