"""

"""
import math
from random import randint

from scipy import sparse
import numpy as np

def dist(a, b):
    """Element-wise distance between to sparce matrix 1xM"""
    if a.shape == b.shape:
        return (a - b).power(2).sum(axis=1)[0, 0]
    else:
        raise ValueError('a and b must have the same shape')


def dist_cosinus():
    pass


# POINT = (question vectorisée, reponse vectorisée)
def composite_distance(point, prototype, x, weights):
    """
    Compute  point-to-prototype (or point-to-point) distance

    :param point, prototype:
    :param x:
    :param weights:
    :return:
    """
    if point.shape == prototype.shape:
        if point.shape[1] % 2 == 0:
            d1 = dist(point[:, 0:int(point.shape[1] / 2)], prototype[:, 0:int(prototype.shape[1] / 2)])
            d2 = dist(point[:, int(point.shape[1] / 2):], prototype[:, int(prototype.shape[1] / 2):])

            temp = 0
            if weights[0] * d1 != 0:
                temp += math.pow(weights[0] * d1, x)
            if weights[1] * d2 != 0:
                temp += math.pow(weights[1] * d2, x)
            return temp
        else:
            raise ValueError('Length of vectors must be even')  # by construction
    else:
        raise ValueError('point and prototype must have the same shape')


class MixKMeans:
    def __init__(self, x, weights):
        """
        Initialize the MixKMeans model with hyper-parameters used in distance
        computations

        :param x: negative float
        :param weights: tuple or list of the two weights given to each part
        """
        if x != 1 and x > 0:
            raise ValueError('x must be negative or zero')
        else:
            self.x = x  # huge negative

        if weights[0] + weights[1] != 1:
            raise ValueError('Sum of weights must be one')
        else:
            self.weights = weights

        # others objects ?
        self.K = None
        self.itermax = None

        self.prototypes = None  # best prototypes

    # --------------
    # - Methods needed to fit the model to the data
    # --------------
    def initialize_prototypes(self, dataset, K):
        """
        Initialize prototypes (i.e. centroid in this mixkmeans) inspired to 'spreading out the cluster centroids'
        :param dataset:
        :param K: number of clusters
        """
        indexes = [randint(0, dataset.shape[0] - 1)]

        for i in range(K - 1):
            vec_dist = []
            for row in dataset:
                sum_dist = 0
                for ind in indexes:
                    sum_dist += composite_distance(row, dataset[ind], self.x, self.weights)
                vec_dist.append(sum_dist)

            indexes.append(np.argmax(vec_dist))

        prototypes = []
        for ind in indexes:
            prototypes.append(dataset[ind])
        self.prototypes = prototypes

    def assign_clusters(self, dataset):
        """
        Assign each point of the dataset to a cluster defines by its prototype

        :param dataset:
        :param prototypes: list of indexes of prototypes
        :return:
        """
        assignation = []
        for row in dataset:
            distances = []
            for index, prototype in enumerate(self.prototypes):
                distances.append(composite_distance(row, prototype, self.x, self.weights))
            assignation.append(np.argmin(distances))

        return assignation

    def compute_prototypes(self, dataset, assignation):
        """
        Compute prototypes for each cluster
        :return:
        """
        prototypes = []
        for cluster_ind in range(self.K):

            indexes = np.where(np.array(assignation) == cluster_ind)[0]  # indexes where Q | A in current cluster
            Q = dataset[indexes, 0:int(dataset.shape[1] / 2)]
            A = dataset[indexes, int(dataset.shape[1] / 2):]

            sum_dist_q = 0.00001
            sum_dist_a = 0.00001
            print(dataset[indexes].shape)

            if dataset[indexes].shape[0] == 0:
                prototypes.append(None)
                continue

            for index, row in enumerate(dataset[indexes]):
                c_d = composite_distance(row, self.prototypes[cluster_ind], self.x, self.weights)
                if c_d != 0:
                    distance_qa = math.pow(c_d, (1 - self.x)/self.x) # noqa
                else:
                    distance_qa = 0
                dist_q = distance_qa * composite_distance(row, self.prototypes[cluster_ind], self.x - 1, (1, 0))
                dist_a = distance_qa * composite_distance(row, self.prototypes[cluster_ind], self.x - 1, (0, 1))

                Q[index] = Q[index].multiply(dist_q)
                A[index] = A[index].multiply(dist_a)

                sum_dist_q += dist_q
                sum_dist_a += dist_a

            Q = Q.multiply(1 / sum_dist_q)
            A = A.multiply(1 / sum_dist_a)

            qa = np.concatenate([np.array(Q.sum(axis=0)), np.array(A.sum(axis=0))], axis=1)
            prototypes.append(sparse.csr_matrix(qa))

        self.prototypes = prototypes

    # --------------
    # - fit and predict
    # --------------

    def fit(self, dataset, K, itermax):
        """
        Process training of MixKmeans model on our dataset

        :param dataset: a sparse matrix with Q | A in rows
        :param K: number of clusters
        :return:
        """
        # print('Begin fitting')
        self.K = K
        self.itermax = itermax
        self.initialize_prototypes(dataset, self.K)

        iteration = 0
        cost = 0
        condition = True
        while (iteration < self.itermax) & condition:
            print(iteration)
            # assigner à chq point un cluster
            assignation = self.assign_clusters(dataset)
            old_prototypes = self.prototypes
            self.compute_prototypes(dataset, assignation)

            cost = 0
            for ind, prototype in enumerate(old_prototypes):
                '''
                print(10*'---')
                print(self.prototypes[ind].shape)
                print(prototype.shape)'''
                c_d = composite_distance(self.prototypes[ind], prototype, self.x, self.weights)
                if c_d != 0:
                    cost += math.pow(c_d, 1 / self.x)
            condition = (cost >= 0.0001)
            iteration += 1

        # message pour dire qu'il n'y a pas eu convergence
        if iteration >= self.itermax:
            print('Pas de convergence ! Processus arrêté au bout de {} iterations)'.format(iteration))  # english

        last_assignation = self.assign_clusters(dataset)

        return self.prototypes, last_assignation, cost

    # Dataset ou moins # TODO
    def predict(self, dataset):
        if self.prototypes:
            assignation = self.assign_clusters(dataset)
            return assignation
        else:
            raise TypeError('Need to fit the model before')


if __name__ == '__main__':
    pass
