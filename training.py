"""
    launch locally (jupyter notebook)
"""
from scipy import sparse

import numpy as np

from mixkmeans import MixKMeans


def launch(data, distance, save_model, save_historic):
    dtm = sparse.load_npz(data)

    COST_HISTORIC = []
    for K in np.arange(50, 550, 50):
        print('begin fitting with {} clusters'.format(K))
        model = MixKMeans(x=-3, weights=(0.2, 0.8), distance=distance, save_file=save_model)
        cost = model.fit(dtm, K, 50)
        COST_HISTORIC.append(cost)

        np.savez(save_historic, K, COST_HISTORIC)


if __name__ == '__main__':
    pass
