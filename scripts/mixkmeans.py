"""

"""
import numpy as np


def compute_prototypes():
    """
    Compute prototypes for each cluster
    :return:
    """
    pass


def assign_clusters(dataset, prototypes):
    """
    Assign each point of the dataset to a cluster defines by its prototype

    :param dataset:
    :param prototypes:
    :return:
    """
    pass


# TODO : format de a et b ?!?!?!!!
def dist(a, b):
    """Element-wise distance"""
    # PROBABLEMENT PAS NUMPY
    return sum((a - b)**2)


def composite_distance(point, prototype, x, weights, dtype):
    """
    Compute  point-to-prototype (or point-to-point) distance

    :param point, prototype:
    :param x:
    :param weights:
    :param dtype:
    :return:
    """
    dic = {'qa': (1, 1), 'q': (1, 0), 'a': (0, 1)}
    d1 = pow(dist(point[0], prototype[0]), x)
    d2 = pow(dist(point[1], prototype[1]), x)
    return dic[dtype][0] * weights[0] * d1 + dic[dtype][1] * weights[1] * d2


class MixKMeans:
    def __init__(self, x, weights):
        """
        Initialize the MixKMeans model with hyper-parameters used in distance
        computations

        :param x: negative float
        :param weights: tuple or list of the two weights given to each part
        """
        if x > 0:
            raise ValueError('x must be negative or zero')
        else:
            self.x = x  # huge negative

        if weights[0] + weights[1] != 1:
            raise ValueError('Sum of weights must be one')
        else:
            self.v = weights[0]
            self.w = weights[1]

        # others objects ?
        self.K = None
        self.itermax = None

        self.prototypes = None  # best prototypes

    def fit(self, dataset, K=4, itermax=100):
        """
        Process training of MixKmeans model on our dataset

        :param dataset:
        :param K: number of clusters
        :return:
        """
        self.K = K
        self.itermax = itermax

        # Initialization
            # choix d'un point aléatoire

            # jusqu'à 3 autre choix de point faire choix du point le plus éloigné des points précédents  # noqa

        iteration = 0
        condition = True
        while iteration < self.itermax & condition:
            # assigner à chq point un cluster

            # estimer les cluster prototypes


            # condition =  sur la distance et donc l'erreur
            iteration += 1

        # sauvegarder les prototypes dans self.prototypes
        print('Done ! (in {} iterations)'.format(iteration))

    def predict(self, dataset):
        if self.prototypes:
            # assigner chaque point à un cluster

            # retourner l'affectation de chaque clusters
            pass
        else:
            raise TypeError('')


if __name__ == '__main__':
    pass
