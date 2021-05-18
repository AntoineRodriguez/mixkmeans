"""
    Code pour Google colaboratory
"""
import pickle
import argparse
from scipy import sparse

import numpy as np

from mixkmeans import MixKMeans

parser = argparse.ArgumentParser()
parser.add_argument("data", help="path to data file (.npz)")
parser.add_argument("-d", "--distance", help="used distance (euclidean or cosinus)")
parser.add_argument("-s", "--save", help="path to save the MixKmeans object if an error happen")

data, dist, save = parser.parse_args()

if __name__ == '__main__':

    # Chargement
    dtm = sparse.load_npz(data)

    COST_HISTORIC = []
    for K in np.arange(50, 550, 50):
        print('begin fitting with {} clusters'.format(K))
        model = MixKMeans(x=-3, weights=(0.2, 0.8), distance=dist, save_file=save)
        cost = model.fit(dtm, K, 50)
        COST_HISTORIC.append(cost)

        np.savez('', K, COST_HISTORIC)
