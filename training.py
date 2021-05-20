"""
    launch locally (jupyter notebook)
"""
from scipy import sparse

import numpy as np

from mixkmeans import MixKMeans


def launch(data, distance, save_model, save_historic, K, number, min_cost=1000000000000000000000000000000000000000000):
    dtm = sparse.load_npz(data)

    COST_HISTORIC = []
    for item in range(number):
        print('begin fitting with {} clusters'.format(K))
        print('initialization number {}'.format(item + 1))
        model = MixKMeans(x=-3, weights=(0.2, 0.8), distance=distance, save_file=save_model)
        cost = model.fit(dtm, K, 50)

        if cost <= min_cost:
            model.save_state()
            min_cost = cost

        COST_HISTORIC.append(cost)

        np.savez(save_historic, K, COST_HISTORIC)


if __name__ == '__main__':
    pass
