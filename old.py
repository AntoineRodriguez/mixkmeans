def _initialize_prototypes(self, dataset, K):
    """Initialize prototypes (i.e. centroid in this mixkmeans) inspired to 'spreading out the cluster centroids'
    :param dataset:
    :param K: number of clusters
    """
    self.K = K
    prototype_ind = randint(0, dataset.shape[0] - 1)
    dist_vec = []  # distance vector that permit to choose the farthest elements in dataset
    for row in dataset:
        dist_vec.append(composite_distance(row, dataset[prototype_ind], self.x, self.weights))
    dist_vec = np.array(dist_vec)

    prototypes = [dataset[prototype_ind]]
    for i in range(self.K - 1):
        prototype_ind = np.argmax(dist_vec)
        print(prototype_ind)
        temp = []
        for row in dataset:
            temp.append(composite_distance(row, dataset[prototype_ind], self.x, self.weights))

        temp = np.array(temp)
        dist_vec += temp
        prototypes.append(dataset[prototype_ind])

    self.prototypes = prototypes


def _compute_prototypes(self, dataset, assignation):
    """
    Compute prototypes for each cluster
    :return:
    """
    prototypes = []
    for cluster_ind in range(self.K):
        vec_Q = []
        vec_A = []
        indexes = np.where(np.array(assignation) == cluster_ind)[0]  # indexes where Q | A in current cluster
        for index in indexes:
            # construct vector for dot product to compute (6) and (7) from Deepak S
            distance_qa = composite_distance(dataset[index], self.prototypes[cluster_ind], self.x, self.weights)
            if distance_qa != 0:
                vec_Q.append(math.pow(distance_qa, (1 - self.x) / self.x) * composite_distance(dataset[index],
                                                                                               self.prototypes[
                                                                                                   cluster_ind],
                                                                                               self.x - 1,
                                                                                               (1, 0)))  # noqa
                vec_A.append(math.pow(distance_qa, (1 - self.x) / self.x) * composite_distance(dataset[index],
                                                                                               self.prototypes[
                                                                                                   cluster_ind],
                                                                                               self.x - 1,
                                                                                               (0, 1)))  # noqa
            else:
                vec_Q.append(0)
                vec_A.append(0)

        # dot products
        print(np.array([vec_Q]).shape)
        Q = dataset[indexes, 0:int(dataset.shape[1] / 2)]
        print(Q.shape)
        q = (np.array([vec_Q]) * sum(vec_Q)).dot(Q)
        print(q.shape)
        q = list(np.array(q.sum(axis=0))[0])

        del Q
        A = dataset[indexes, 0:int(dataset.shape[1] / 2)]
        a = (np.array([vec_A]) * sum(vec_A)).dot(A)
        a = list(np.array(a.sum(axis=0))[0])
        del A
        prototypes.append(np.array(q + a))

    self.prototypes = prototypes